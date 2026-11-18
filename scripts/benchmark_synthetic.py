"""Sintetik stress-benchmark: N ta tasodifiy javon (standart 100) — tanish + OOS.

Har javon: 14 SKU (rang-egizaklar va shakl-egizaklar bilan), ~15% katalogda
yo'q mahsulot, bo'sh kataklar, real kamera buzilishlari (yorug'lik, oq balans,
blur, shovqin, JPEG). Katalog esa toza "packshot"lardan — ya'ni domen farqi bor.

Bir xil javonlarda bir nechta konfiguratsiya solishtiriladi:
  legacy  — eski tizim (markaziy kesish, faqat CLS, rangsiz, kontekstsiz)
  new     — config.yaml dagi joriy sozlamalar
va --set orqali istalgan boshqa variant.

Metrikalar:
  tanish: umumiy aniqlik, tanish mahsulotlar aniqligi, noma'lumni rad etish
  OOS:    recall (bo'sh kataklarning necha foizi topildi), precision

  python scripts/benchmark_synthetic.py --n 100
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401

import numpy as np

from rasta import synthetic as S
from rasta.calibrate import Rec, report
from rasta.catalog import RecognitionParams, SkuCatalog
from rasta.config import ROOT, Config

LEGACY = {
    "embedder.model": "facebook/dinov2-base",
    "embedder.weights": None,
    "embedder.resize_mode": "crop",
    "embedder.pooling": "cls",
    "recognition.w_color": 0.0,
    "recognition.w_shape": 0.0,
    "recognition.color_min": 0.0,
    "context.enabled": False,
    "oos.use_shelf_extent": False,
    "oos.verify": False,
    "oos.empty_shelves": False,
    "oos.require_frontal": False,
    "shelf.anchor": "center",
}


class FixedDetector:
    def __init__(self):
        self.boxes = []

    def detect(self, image):
        return list(self.boxes)


def oos_metrics(gaps, empties_xy):
    """Bo'sh katak markazlari gap qutilari ichida bo'lsa — topilgan."""
    covered = 0
    for (x, y) in empties_xy:
        if any(g.x1 <= x <= g.x2 and g.y1 - 5 <= y <= g.y2 + 5 for g in gaps):
            covered += 1
    tp_gaps = sum(1 for g in gaps if any(g.x1 <= x <= g.x2 and g.y1 - 5 <= y <= g.y2 + 5
                                         for (x, y) in empties_xy))
    return covered, len(empties_xy), tp_gaps, len(gaps)


def slot_center(r, c, cell_w=150, cell_h=320, gap=14, margin=30, shelf_thick=16):
    y0 = margin + r * (cell_h + gap + shelf_thick)
    x0 = margin + c * (cell_w + gap)
    return x0 + cell_w / 2, y0 + cell_h * 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Javonlar soni")
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--strength", type=float, default=1.0, help="Buzilish kuchi")
    ap.add_argument("--unknown-frac", type=float, default=0.15)
    ap.add_argument("--set", action="append", default=[],
                    help="'new' konfiguratsiyaga o'zgartirish a.b=v")
    ap.add_argument("--ablation", action="store_true",
                    help="Komponentlarni birma-bir qo'shib solishtirish")
    ap.add_argument("--out", default="outputs/benchmark_synthetic.json")
    args = ap.parse_args()

    from rasta.pipeline import RastaPipeline

    base = Config.load().override(args.set)
    base.set("catalog.path", ":memory:")
    base.set("detector.backend", "simple")
    pipe = RastaPipeline(base)
    det = FixedDetector()
    pipe.detector = det

    legacy = base.copy().override([f"{k}={v}" for k, v in LEGACY.items()])
    configs = {"legacy": legacy, "new": base.copy()}
    legacy.set("embedder.weights", None)
    if args.ablation:
        steps = [
            ("A_pad", ["embedder.resize_mode=pad"]),
            ("A_stretch", ["embedder.resize_mode=stretch"]),
            ("B_stretch_clsmean", ["embedder.resize_mode=stretch", "embedder.pooling=cls+mean"]),
            ("C_color0.15", ["recognition.w_color=0.15"]),
            ("C_color0.3", ["recognition.w_color=0.3"]),
            ("C_color0.45", ["recognition.w_color=0.45"]),
            ("D_color0.3_shape0.1", ["recognition.w_color=0.3", "recognition.w_shape=0.1"]),
            ("E_color0.3_veto0.5", ["recognition.w_color=0.3", "recognition.color_min=0.5"]),
            ("F_color0.3_ctx", ["recognition.w_color=0.3", "context.enabled=true"]),
        ]
        cum = ["embedder.resize_mode=stretch", "embedder.pooling=cls+mean"]
        steps += [("G_siglip2", ["embedder.model=google/siglip2-base-patch16-224",
                                 "embedder.resize_mode=stretch"]),
                  ("H_siglip2_color0.2_shape0.05", ["embedder.model=google/siglip2-base-patch16-224",
                                                     "embedder.resize_mode=stretch",
                                                     "recognition.w_color=0.2",
                                                     "recognition.w_shape=0.05"])]
        configs = {"legacy": legacy}
        for name, ov in steps:
            ov2 = ov if name.startswith(("A_", "B_", "G_", "H_")) else cum + ov
            configs[name] = legacy.copy().override(ov2)
        configs["new"] = base.copy()

    # Javonlar bir marta yaratiladi — barcha konfiguratsiyalar bir xil ma'lumotda
    shelves = []
    for i in range(args.n):
        rng = np.random.default_rng(i)
        img, boxes, labels, empties = S.generate_shelf(
            rows=3, cols=8, empty_slots=int(rng.integers(2, 7)), seed=1000 + i,
            products=S.ALL_PRODUCTS, unknown_frac=args.unknown_frac,
            block_size=int(rng.integers(1, 4)), return_empties=True)
        img = S.distort(img, seed=i, strength=args.strength)
        shelves.append((img, boxes, labels, [slot_center(r, c) for r, c in empties]))

    catalog_imgs = S.generate_catalog(variants=args.variants, products=S.ALL_PRODUCTS)
    name_of = {p.sku_id: p.name for p in S.ALL_PRODUCTS}

    summary = {}
    for cname, cfg in configs.items():
        t0 = time.time()
        pipe.cfg = cfg
        pipe.params = RecognitionParams.from_cfg(cfg)
        if cfg.get("embedder.model") != pipe.embedder.model_name:
            from rasta.embedder import build_embedder
            pipe.embedder = build_embedder(cfg, pipe.device)
        pipe.embedder.resize_mode = cfg.get("embedder.resize_mode")
        pipe.embedder.pooling = cfg.get("embedder.pooling")
        pipe.embedder._dim = None
        pipe.catalog = SkuCatalog(pipe.embedder.dim, client=pipe.catalog.client,
                                  collection=f"syn_{cname}")
        pipe.catalog.reset()
        for sku, imgs in catalog_imgs.items():
            pipe.add_product(sku, name_of[sku], imgs, brand=S.BRAND_OF_ALL[sku], source="packshot")

        recs, groups = [], []
        tot = ok = kn = kn_ok = un = un_ok = 0
        cov = emp = tpg = ng = 0
        per_shelf = []
        confusion = {}
        for si, (img, boxes, labels, empties_xy) in enumerate(shelves):
            det.boxes = boxes
            res = pipe.analyze_image(img)
            # detection'lar tartibi box'lar bilan bir xil (fon qatori bo'lmasa)
            by_box = {(d.box.x1, d.box.y1): d for d in res.detections}
            s_ok = 0
            for b, lab in zip(boxes, labels):
                d = by_box.get((b.x1, b.y1))
                pred = d.match.sku_id if (d and d.match and d.match.is_known) else "unknown"
                truth = "unknown" if lab.startswith("unknown:") else lab
                if d and d.match and d.match.candidates:
                    recs.append(Rec(truth, d.match.candidates[0]["sku_id"],
                                    d.match.score, d.match.margin))
                    groups.append(si)
                tot += 1
                hit = pred == truth
                ok += hit
                s_ok += hit
                if truth == "unknown":
                    un += 1; un_ok += hit
                else:
                    kn += 1; kn_ok += hit
                if not hit:
                    key = f"{truth}->{pred}"
                    confusion[key] = confusion.get(key, 0) + 1
            c, e, tp, g = oos_metrics(res.gaps, empties_xy)
            cov += c; emp += e; tpg += tp; ng += g
            per_shelf.append(s_ok / len(labels))
        summary[cname] = {
            "shelves": len(shelves),
            "products": tot,
            "accuracy": round(ok / tot, 4),
            "known_acc": round(kn_ok / kn, 4) if kn else None,
            "unknown_reject": round(un_ok / un, 4) if un else None,
            "worst_shelf_acc": round(min(per_shelf), 4),
            "oos_recall": round(cov / emp, 4) if emp else None,
            "oos_precision": round(tpg / ng, 4) if ng else None,
            "calibration": report(recs, groups),
            "top_confusions": dict(sorted(confusion.items(), key=lambda kv: -kv[1])[:8]),
            "seconds": round(time.time() - t0, 1),
        }
        print(cname, json.dumps(summary[cname], ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "results": summary}, ensure_ascii=False,
                              indent=1), encoding="utf-8")
    print(f"[✓] {out}")


if __name__ == "__main__":
    main()
