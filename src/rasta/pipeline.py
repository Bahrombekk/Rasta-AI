"""To'liq tahlil pipeline'i — rasmni AnalysisResult'ga aylantiradi.

Qadamlar (hisobotning 6.1-bo'limi + kuchaytirishlar):
  1. Detektsiya (class-agnostic) + ichma-ich qutilarni tozalash
  2. Javon qatorlari (pastki chiziq bo'yicha) + orqa fon qatorlarini ajratish
  3. Embedding (DINOv2) + rang va shakl deskriptorlari
  4. Gibrid vektor qidiruv (SKU tanish, open-set, rang veto'si)
  5. Javon konteksti (qo'shnilar bo'yicha noma'lumlarni tuzatish)
  6. Noma'lumlarni klasterlash (HDBSCAN)
  7. OOS (bo'sh joy) + rasm bo'yicha tekshiruv
  8. KPI
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .catalog import RecognitionParams, build_catalog
from .config import Config, resolve_device
from .context import smooth_with_context
from .detector import build_detector, suppress_nested
from .embedder import build_embedder, crop_boxes
from .features import aspect_ratio, color_descriptors, to_rgb_array
from .imageio import load_image
from .kpi import compute_kpi
from .oos import detect_gaps, verify_gaps
from .shelf import assign_rows, mark_background_rows, renumber_rows
from .types import AnalysisResult, Detection


class RastaPipeline:
    def __init__(self, cfg: Optional[Config] = None, *, inference=None):
        self.cfg = cfg or Config.load()
        if inference is None:
            self.device = resolve_device(self.cfg.get("device", "auto"))
            self.detector = build_detector(self.cfg, self.device)
            self.embedder = build_embedder(self.cfg, self.device)
        else:
            self.device = inference.device
            self.detector = self.embedder = inference
        self.catalog = build_catalog(self.cfg, dim=self.embedder.dim)
        self.params = RecognitionParams.from_cfg(self.cfg)

    # ---- Katalog onboarding ----------------------------------------------

    def embed_images(self, images: list) -> np.ndarray:
        return self.embedder.embed(images)

    def describe_images(self, images: list) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Rasmlar -> (DINO vektorlar, rang deskriptorlari, tomonlar nisbati)."""
        arrs = [to_rgb_array(im) for im in images]
        vecs = self.embedder.embed(arrs)
        cols = color_descriptors(arrs)
        asps = [aspect_ratio(a.shape[0], a.shape[1]) for a in arrs]
        return vecs, cols, asps

    def add_product(self, sku_id: str, name: str, images: list,
                    brand: Optional[str] = None, extra: Optional[dict] = None,
                    augment: Optional[bool] = None, source: Optional[str] = None) -> int:
        """Few-shot onboarding: rasmlar -> katalog (embedding + rang + shakl).

        Rasm kam bo'lsa (onboarding.augment_if_fewer dan kam — standart: faqat
        bitta packshot), undan onboarding.augment ta realistik variant yaratiladi
        (Grocery benchmark: 1 packshot -> 16 variant, closed-set 0.815 -> 0.896).
        augment=False — real javon crop'lari uchun: ularni augmentatsiya qilish
        katalogni "yopishqoq" qiladi (drinks_aisle'da 29 ta noto'g'ri tanish).
        source: "packshot" | "crop" — qaysi chegara (τ) qo'llanishini belgilaydi;
        berilmasa: augmentatsiya qilingan yolg'iz rasm -> packshot, aks holda crop.
        """
        # Bir xil (oq) fonli katalog rasmlari mahsulot chegarasigacha kesiladi va
        # packshot deb belgilanadi (internet rasmlarida mahsulot ko'pincha kichik:
        # real muzlatgich sinovida o'xshashlik 0.60 -> 0.82, tanish 0/27 -> 24/27).
        if source != "crop" and self.cfg.get("onboarding.packshot_crop", True):
            from .augment import packshot_crop
            res = [packshot_crop(im) for im in images]
            images = [r[0] for r in res]
            if source is None and sum(r[1] for r in res) * 2 > len(res):
                source = "packshot"
        n_aug = int(self.cfg.get("onboarding.augment", 16) or 0)
        few = len(images) < int(self.cfg.get("onboarding.augment_if_fewer", 2))
        augmented = augment is not False and n_aug > 1 and (augment or few or source == "packshot")
        if augmented:
            from .augment import make_variants
            images = [v for i, im in enumerate(images)
                      for v in make_variants(im, n_aug, seed=i)]
        vecs, cols, asps = self.describe_images(images)
        payload = dict(extra or {})
        if brand:
            payload["brand"] = brand
        payload["src"] = source or ("packshot" if augmented else "crop")
        return self.catalog.add_sku(sku_id, name, vecs, extra=payload or None,
                                    colors=cols, aspects=asps)

    # ---- Tahlil ------------------------------------------------------------

    def analyze_image(
        self,
        image: np.ndarray,
        image_path: str = "<array>",
        brand_of: Optional[dict] = None,
        detect_image: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None,
        assume_frontal: bool = False,
    ) -> AnalysisResult:
        """detect_image — detektor uchun alternativ rasm (masalan poligon tashqarisi
        yopilgan); roi_mask — faqat shu maska ichidagi mahsulot/bo'shliqlar qoladi."""
        cfg = self.cfg
        h, w = image.shape[:2]
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        # 1. Detektsiya
        boxes = self.detector.detect(detect_image if detect_image is not None else image)
        if cfg.get("detector.suppress_nested", True):
            boxes = suppress_nested(boxes, cfg.get("detector.contain_thr", 0.8))
        if roi_mask is not None:
            from .roi import inside_fraction
            min_in = cfg.get("roi.min_inside", 0.5)
            boxes = [b for b in boxes if inside_fraction(b, roi_mask) >= min_in]
        detections = [Detection(box=b) for b in boxes]
        t["detect"] = (time.perf_counter() - t0) * 1000

        # 2. Javon qatorlari + orqa fon
        t1 = time.perf_counter()
        num_rows = assign_rows(
            detections,
            row_tolerance=cfg.get("shelf.row_tolerance", 0.6),
            anchor=cfg.get("shelf.anchor", "bottom"),
        )
        background: list[Detection] = []
        if cfg.get("shelf.drop_background_rows", True) and num_rows > 1:
            bg_rows = mark_background_rows(detections, num_rows)
            if bg_rows:
                background = [d for d in detections if d.row in bg_rows]
                detections = [d for d in detections if d.row not in bg_rows]
                num_rows = renumber_rows(detections)
        t["rows"] = (time.perf_counter() - t1) * 1000

        # 3 + 4. Embedding + gibrid tanish
        t2 = time.perf_counter()
        vectors = colors = None
        aspects: list[float] = []
        if detections:
            crops = crop_boxes(image, [d.box for d in detections])
            vectors = self.embedder.embed(crops)
            for d, v in zip(detections, vectors):
                d.vec = v
            colors = color_descriptors(crops)
            aspects = [aspect_ratio(d.box.height, d.box.width) for d in detections]
            if self.catalog.count() > 0:
                matches = self.catalog.search(vectors, self.params, colors, aspects)
                for d, m in zip(detections, matches):
                    d.match = m
        t["recognize"] = (time.perf_counter() - t2) * 1000

        # 5. Javon konteksti
        if cfg.get("context.enabled", True) and detections:
            smooth_with_context(
                detections,
                sim_threshold=self.params.sim_threshold,
                relax=cfg.get("context.relax", 0.08),
                color_min=self.params.color_min,
                params=self.params,
            )
            if cfg.get("context.propagate_blocks", False):
                from .context import propagate_identical
                propagate_identical(
                    detections,
                    sim=cfg.get("context.block_sim", 0.90),
                    max_gap=cfg.get("context.block_max_gap", 0.6),
                    params=self.params,
                )

        # 6. Noma'lumlarni klasterlash
        clusters: dict = {}
        if cfg.get("clustering.enabled", True) and vectors is not None:
            clusters = self._cluster_unknowns(detections, vectors, colors, aspects)

        # 7. OOS
        t3 = time.perf_counter()
        gaps, dividers = detect_gaps(
            detections,
            num_rows,
            min_gap_ratio=cfg.get("oos.min_gap_ratio", 0.8),
            edge_margin=cfg.get("oos.edge_margin", 0.02),
            divider_row_frac=cfg.get("oos.divider_row_frac", 0.6),
            use_shelf_extent=cfg.get("oos.use_shelf_extent", True),
            return_dividers=True,
        )
        suspects = []
        if cfg.get("oos.verify", True):
            gaps, suspects = verify_gaps(
                image, gaps, detections, suspect_ratio=cfg.get("oos.suspect_ratio", 0.55))
        # Sahna tekshiruvi: frontal javonmi? (mahsulotlar javon chetida turadimi)
        scene_support = None
        if detections and (cfg.get("oos.empty_shelves", True) or cfg.get("oos.require_frontal", True)):
            from .shelf_lines import detect_shelf_edges, find_empty_shelves, shelf_support
            edges = detect_shelf_edges(image)
            scene_support = shelf_support(edges, detections)
            if cfg.get("oos.require_frontal", True) and not assume_frontal and                     scene_support < cfg.get("oos.frontal_min_support", 0.8):
                # Yo'lak/ombor/qiya sahna: "qator" geometriyasi ishonchsiz — OOS emas, shubhali
                for g in gaps:
                    g.status = "suspect"
                suspects = suspects + gaps
                gaps = []
            # Butunlay bo'sh javon zonalari (javon chetlari bo'yicha)
            if cfg.get("oos.empty_shelves", True):
                shelf_gaps, _ = find_empty_shelves(
                    image, detections, existing=gaps + suspects + dividers, edges=edges,
                    min_support_frac=cfg.get("oos.shelf_support_frac", 0.85))
                gaps = gaps + shelf_gaps
        t["oos"] = (time.perf_counter() - t3) * 1000

        if roi_mask is not None:
            # zona tashqarisidagi bo'shliqlar hisobga olinmaydi
            def _in(g):
                cx, cy = int((g.x1 + g.x2) / 2), int((g.y1 + g.y2) / 2)
                return 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]                     and bool(roi_mask[cy, cx])
            gaps = [g for g in gaps if _in(g)]
            suspects = [g for g in suspects if _in(g)]
            dividers = [g for g in dividers if _in(g)]

        # 8. KPI (brend: argument > katalog payload)
        if brand_of is None:
            brand_of = {d.match.sku_id: d.match.brand for d in detections
                        if d.match and d.match.is_known and d.match.brand} or None
        # Klasterlashga faqat haqiqatan noma'lumlar kiradi (brend darajasidagilar emas)
        kpi = compute_kpi(detections, gaps, num_rows, brand_of=brand_of)
        oos_status = "estimated"
        if not gaps and len(detections) <= num_rows:
            oos_status = "insufficient_products"
        elif (cfg.get("oos.require_frontal", True) and not assume_frontal
              and scene_support is not None and scene_support < cfg.get("oos.frontal_min_support", 0.8)):
            oos_status = "limited_scene"
        kpi["oos"]["status"] = oos_status
        kpi["oos"]["num_suspect"] = len(suspects)
        kpi["oos"]["num_divider"] = len(dividers)
        if roi_mask is not None:
            # To'ldirilganlik: zona maydonining mahsulotlar egallagan ulushi.
            # Siyrak/bo'sh muzlatgichda qator-asosli OOS ishlamaydi, bu esa ishlaydi.
            occ = np.zeros_like(roi_mask)
            for d in detections:
                x1, y1, x2, y2 = (max(0, int(v)) for v in d.box.as_xyxy())
                occ[y1:y2, x1:x2] = True
            area = max(1, int(roi_mask.sum()))
            kpi["roi_fill_pct"] = round(100.0 * float((occ & roi_mask).sum()) / area, 1)
            # Qator bo'yicha: har qatorda mahsulotlar zona kengligining qancha qismini
            # egallaydi (shishalar orasidagi tirqishlar va ramka hisobga olinmaydi).
            row_fill = []
            for r in range(num_rows):
                rd = [d for d in detections if d.row == r]
                if not rd:
                    continue
                y = int(np.clip(np.median([d.box.cy for d in rd]), 0, roi_mask.shape[0] - 1))
                width = int(roi_mask[y].sum())
                if width <= 0:
                    continue
                covered = np.zeros(roi_mask.shape[1], dtype=bool)
                med_w = float(np.median([d.box.width for d in rd]))
                for d in rd:   # qo'shni mahsulotlar orasidagi tor tirqish (<25% kenglik) to'ldiriladi
                    x1 = max(0, int(d.box.x1 - 0.125 * med_w))
                    x2 = int(d.box.x2 + 0.125 * med_w)
                    covered[x1:x2] = True
                row_fill.append(float((covered & roi_mask[y]).sum()) / width)
            kpi["roi_row_fill_pct"] = [round(100 * v, 1) for v in row_fill]
            kpi["roi_shelf_fill_pct"] = round(100 * float(np.mean(row_fill)), 1) if row_fill else 0.0
        if scene_support is not None:
            kpi["scene_shelf_support"] = round(scene_support, 3)
            kpi["scene_frontal"] = assume_frontal or                 scene_support >= cfg.get("oos.frontal_min_support", 0.8)
        kpi["num_unknown_clusters"] = len([k for k in clusters if k != "noise"])
        t["total"] = (time.perf_counter() - t0) * 1000

        return AnalysisResult(
            image_path=image_path,
            image_size=(w, h),
            detections=detections,
            gaps=gaps,
            rows=num_rows,
            kpi=kpi,
            suspects=suspects,
            dividers=dividers,
            background=background,
            clusters=clusters,
            timings_ms=t,
        )

    def _cluster_unknowns(self, detections, vectors, colors, aspects) -> dict:
        from .cluster import build_cluster_features, cluster_features, summarize

        idx = [i for i, d in enumerate(detections)
               if not (d.match and (d.match.is_known or d.match.level == "brand"))]
        if len(idx) < 2:
            return {}
        feats = build_cluster_features(
            vectors[idx], colors[idx], np.asarray(aspects)[idx],
            color_weight=self.cfg.get("clustering.color_weight", 0.6),
            shape_weight=self.cfg.get("clustering.shape_weight", 0.3),
        )
        labels = cluster_features(
            feats, min_cluster_size=self.cfg.get("clustering.min_cluster_size", 2))
        for i, lab in zip(idx, labels):
            detections[i].cluster = int(lab)
        return summarize(labels, idx)

    def analyze_region(self, image: np.ndarray, region, image_path: str = "<array>",
                       brand_of: Optional[dict] = None) -> AnalysisResult:
        """Faqat poligon (roi.Region) ichidagi javon/muzlatgichni tahlil qiladi.

        Zona kesiladi va kattalashtiriladi (roi.target_long), detektor tashqarisi
        yopilgan rasmni ko'radi, embedding esa asl piksellardan olinadi. Natija
        koordinatalari asl rasmga qaytariladi.
        """
        from . import roi as R

        work = R.prepare(image, region,
                         target_long=self.cfg.get("roi.target_long", 1280),
                         max_scale=self.cfg.get("roi.max_scale", 3.0),
                         rectify=self.cfg.get("roi.rectify", False))
        res = self.analyze_image(work.image, image_path=image_path, brand_of=brand_of,
                                 detect_image=work.detect_image, roi_mask=work.mask,
                                 # 4 burchak bo'yicha to'g'rilangan zona — ta'rifiga ko'ra frontal
                                 assume_frontal=work.Hinv is not None)
        h, w = image.shape[:2]
        return R.to_original(res, work, region, (w, h), image_path)

    def analyze_regions(self, image: np.ndarray, regions, image_path: str = "<array>",
                        brand_of: Optional[dict] = None) -> list[AnalysisResult]:
        return [self.analyze_region(image, r, image_path, brand_of) for r in regions]

    def analyze_path(self, path: str | Path, brand_of: Optional[dict] = None) -> AnalysisResult:
        image = load_image(path)
        return self.analyze_image(image, image_path=str(path), brand_of=brand_of)
