"""Class-agnostic detektor — javondagi barcha mahsulot qutilarini topadi.

Ikki backend:
  - YoloDetector: ultralytics YOLO (real rasmlar uchun; SKU-110K fine-tune tavsiya).
  - SimpleDetector: OpenCV kontur asosida (sintetik/demo yoki YOLO bo'lmaganda).
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .types import Box


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Box]:
        """RGB (H,W,3) rasm -> Box ro'yxati."""
        ...


class YoloDetector:
    """Ultralytics YOLO asosidagi class-agnostic detektor.

    Ixtiyoriy kuchaytirishlar (o'qitishsiz, faqat inference):
      scales — bir nechta imgsz'da o'tish (mayda va yirik mahsulotlar uchun)
      flip   — gorizontal aks ettirilgan rasmda ham o'tish
      tiles  — rasmni N×N bir-birini qoplaydigan bo'laklarga bo'lib o'tish (mayda
               mahsulotlar kattaroq ko'rinadi; 800 px li agent rasmlari uchun)
    Bir nechta o'tish natijasi WBF (weighted box fusion) bilan birlashtiriladi.
    """

    def __init__(
        self,
        weights: str = "yolo11m.pt",
        device: str = "cuda",
        conf: float = 0.20,
        iou: float = 0.55,
        imgsz: int = 1280,
        max_det: int = 600,
        agnostic_nms: bool = True,
        classes: list[int] | None = None,
        scales: list[int] | None = None,
        flip: bool = False,
        tiles: int = 0,
        tile_overlap: float = 0.25,
        tile_imgsz: int | None = None,
        fuse_iou: float = 0.55,
        min_votes: int = 1,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.classes = classes
        self.scales = list(scales) if scales else [imgsz]
        self.flip = flip
        self.tiles = int(tiles or 0)
        self.tile_overlap = tile_overlap
        self.tile_imgsz = tile_imgsz
        self.fuse_iou = fuse_iou
        self.min_votes = min_votes

    def _predict(self, images: list[np.ndarray], imgsz: int) -> list[np.ndarray]:
        """Rasmlar -> har biri uchun (N,5) massiv [x1,y1,x2,y2,score]."""
        results = self.model.predict(
            source=images,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=imgsz,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            classes=self.classes,
            verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(np.zeros((0, 5), dtype=np.float32))
                continue
            out.append(np.concatenate([r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy()[:, None]], axis=1))
        return out

    def passes(self, image: np.ndarray) -> list[np.ndarray]:
        """Har bir o'tish (masshtab / aks / bo'lak to'plami) uchun asl rasm
        koordinatalaridagi qutilar."""
        h, w = image.shape[:2]
        out: list[np.ndarray] = []
        for s in self.scales:
            out.append(self._predict([image], s)[0])
            if self.flip:
                p = self._predict([np.ascontiguousarray(image[:, ::-1])], s)[0]
                p[:, [0, 2]] = w - p[:, [2, 0]]
                out.append(p)
        if self.tiles > 1:
            windows = tile_windows(w, h, self.tiles, self.tile_overlap)
            crops = [np.ascontiguousarray(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in windows]
            tsz = self.tile_imgsz or self.scales[0]
            merged = []
            for (x1, y1, x2, y2), p in zip(windows, self._predict(crops, tsz)):
                p = p[~cut_by_tile(p, x2 - x1, y2 - y1, x1 > 0, y1 > 0, x2 < w, y2 < h)]
                p[:, [0, 2]] += x1
                p[:, [1, 3]] += y1
                merged.append(p)
            # Bo'laklar qoplanadi: bitta mahsulot ikki bo'lakda ham topiladi -> NMS
            out.append(nms_array(np.concatenate(merged), self.iou) if merged
                       else np.zeros((0, 5), dtype=np.float32))
        return out

    def detect(self, image: np.ndarray) -> list[Box]:
        ps = self.passes(image)
        arr = ps[0] if len(ps) == 1 else drop_group_boxes(wbf(ps, self.fuse_iou, self.min_votes))
        return [Box(float(a), float(b), float(c), float(d), float(s)) for a, b, c, d, s in arr]


class ZeroShotDetector:
    """Matn so'rovi bilan ishlaydigan (open-vocabulary) detektor, HuggingFace transformers.

    Grounding DINO oilasi (IDEA-Research/grounding-dino-*, MM-Grounding-DINO, LLMDet)
    va OWLv2 (google/owlv2-*). O'qitishsiz — javon datasetisiz ham "bottle. can. ..."
    so'rovi bilan mahsulotlarni topadi; detektorlarni solishtirish va qoralama
    (pseudo-label) tayyorlash uchun.
    """

    def __init__(self, model_id: str, device: str = "cuda",
                 prompts: tuple[str, ...] = ("bottle", "can", "box", "product package"),
                 conf: float = 0.25, text_conf: float = 0.2, iou: float = 0.5,
                 max_area: float = 0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.owl = "owl" in model_id.lower()
        self.prompts = list(prompts)
        self.conf = conf
        self.text_conf = text_conf
        self.iou = iou
        self.max_area = max_area      # rasmning shu ulushidan katta quti — muzlatgich/javon, mahsulot emas

    def detect(self, image: np.ndarray) -> list[Box]:
        h, w = image.shape[:2]
        if self.owl:
            text = [[f"a photo of a {p}" for p in self.prompts]]
        else:
            text = ". ".join(self.prompts) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        if self.owl:
            s = max(h, w)                 # OWLv2 rasmni kvadratga to'ldiradi
            res = self.processor.post_process_object_detection(
                out, threshold=self.conf, target_sizes=[(s, s)])[0]
        else:
            res = self.processor.post_process_grounded_object_detection(
                out, input_ids=inputs["input_ids"], threshold=self.conf,
                text_threshold=self.text_conf, target_sizes=[(h, w)])[0]
        arr = np.concatenate([res["boxes"].float().cpu().numpy(),
                              res["scores"].float().cpu().numpy()[:, None]], axis=1).reshape(-1, 5)
        arr[:, [0, 2]] = arr[:, [0, 2]].clip(0, w)
        arr[:, [1, 3]] = arr[:, [1, 3]].clip(0, h)
        area = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
        arr = nms_array(arr[(area > 0) & (area < self.max_area * w * h)], self.iou)
        return [Box(float(a), float(b), float(c), float(d), float(s)) for a, b, c, d, s in arr]


class SimpleDetector:
    """OpenCV kontur asosidagi zaxira detektor.

    Sintetik demo va YOLO mavjud bo'lmagan holatlar uchun. Real, zich
    javon rasmlarida YoloDetector'dan foydalaning.
    """

    def __init__(
        self,
        min_area_ratio: float = 0.0008,
        max_area_ratio: float = 0.25,
        pad: int = 1,
    ):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.pad = pad

    def detect(self, image: np.ndarray) -> list[Box]:
        import cv2

        h, w = image.shape[:2]
        img_area = float(h * w)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Qirralar -> morfologik yopish -> konturlar
        edges = cv2.Canny(gray, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: list[Box] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < self.min_area_ratio * img_area:
                continue
            if area > self.max_area_ratio * img_area:
                continue
            x1 = max(0, x - self.pad)
            y1 = max(0, y - self.pad)
            x2 = min(w, x + bw + self.pad)
            y2 = min(h, y + bh + self.pad)
            boxes.append(Box(float(x1), float(y1), float(x2), float(y2), 1.0))
        return boxes


def suppress_nested(boxes: list[Box], contain_thr: float = 0.8) -> list[Box]:
    """Boshqa qutining ichida deyarli to'liq joylashgan kichik qutilarni olib tashlaydi.

    Class-agnostic detektor ba'zan bitta mahsulot uchun "butun" va "qism"
    (masalan faqat yorliq) qutilarini beradi — NMS IoU past bo'lgani uchun
    ikkalasi qoladi va facings ikki marta sanaladi.
    """
    if len(boxes) < 2:
        return list(boxes)
    xyxy = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float)
    areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    keep = np.ones(len(boxes), dtype=bool)
    order = np.argsort(areas)  # kichigidan boshlab
    for i in order:
        if not keep[i] or areas[i] <= 0:
            continue
        ix1 = np.maximum(xyxy[i, 0], xyxy[:, 0]); iy1 = np.maximum(xyxy[i, 1], xyxy[:, 1])
        ix2 = np.minimum(xyxy[i, 2], xyxy[:, 2]); iy2 = np.minimum(xyxy[i, 3], xyxy[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        contain = inter / areas[i]
        contain[i] = 0
        bigger = (areas > areas[i]) & keep
        if np.any((contain >= contain_thr) & bigger):
            keep[i] = False
    return [b for b, k in zip(boxes, keep) if k]


def merge_split_boxes(boxes: list[Box], iomin_thr: float = 0.5, x_iou: float = 0.6,
                      max_union: float = 1.10, max_piece: float = 0.90, min_refs: int = 2,
                      partner_thr: float = 0.3) -> list[Box]:
    """Bitta baland mahsulotni yuqori/pastki ikki qutiga bo'lib yuborishni birlashtiradi.

    Class-agnostic detektor baland kartonni ba'zan "tepa qismi" va "pastki qismi" deb
    ikki marta topadi: qutilar kuchli ustma-ust tushadi, lekin biri ikkinchisining ichida
    emas (suppress_nested ushlamaydi) — mahsulot ikki marta sanaladi.

    Faqat ustma-ustlik bo'yicha birlashtirish XATO: muzlatgichda orqa qatordagi shisha
    oldingisi bilan xuddi shunday ustma-ust tushadi (4 ta GT to'plamda recall -0.3..-0.6
    punkt). Farqi — bo'lak TO'LIQ mahsulotdan past: shuning uchun juftlik faqat quyidagi
    hollarda birlashtiriladi:
      - IoMin >= iomin_thr va gorizontal qoplanish (x bo'yicha IoU) >= x_iou;
      - shu balandlik zonasida "toza" (ustma-ust sherigi yo'q) kamida min_refs ta qo'shni
        bor; ularning balandligi (90-pertsentil) — to'liq mahsulot o'lchovi;
      - har ikki bo'lak shu o'lchovning max_piece qismidan past, birlashgani esa
        max_union baravaridan oshmaydi.
    O'lchov (395 rasm, ~27 000 GT quti): recall o'zgarmadi, precision +0.1 punkt; sok
    javonida 5 ta bo'lingan kartonning 5 tasi birlashdi.
    """
    n = len(boxes)
    if n < 3:
        return list(boxes)
    xyxy = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float)
    hgt = xyxy[:, 3] - xyxy[:, 1]
    areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, hgt)
    ix = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2]) - np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    iy = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3]) - np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    inter = np.maximum(0, ix) * np.maximum(0, iy)
    iomin = inter / np.maximum(1e-9, np.minimum(areas[:, None], areas[None, :]))
    np.fill_diagonal(iomin, 0.0)
    clean = iomin.max(axis=1) < partner_thr
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.where(np.triu(iomin >= iomin_thr, 1))):
        union_w = max(xyxy[i, 2], xyxy[j, 2]) - min(xyxy[i, 0], xyxy[j, 0])
        if ix[i, j] / max(union_w, 1e-9) < x_iou:
            continue
        uy1, uy2 = min(xyxy[i, 1], xyxy[j, 1]), max(xyxy[i, 3], xyxy[j, 3])
        overlap = np.minimum(xyxy[:, 3], uy2) - np.maximum(xyxy[:, 1], uy1)
        refs = clean & (overlap >= 0.5 * hgt)
        if refs.sum() < min_refs:
            continue
        full = float(np.percentile(hgt[refs], 90))
        if (uy2 - uy1) > max_union * full or max(hgt[i], hgt[j]) > max_piece * full:
            continue
        parent[find(int(j))] = find(int(i))

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out: list[Box] = []
    for members in groups.values():
        if len(members) == 1:
            out.append(boxes[members[0]])
            continue
        x1 = min(boxes[m].x1 for m in members); y1 = min(boxes[m].y1 for m in members)
        x2 = max(boxes[m].x2 for m in members); y2 = max(boxes[m].y2 for m in members)
        out.append(Box(x1, y1, x2, y2, max(boxes[m].score for m in members)))
    return out


def drop_group_boxes(arr: np.ndarray, contain_thr: float = 0.8, cover_thr: float = 0.7,
                     grid: int = 32) -> np.ndarray:
    """Bir nechta mahsulotni o'rab olgan "guruh" qutilarini olib tashlaydi.

    Past masshtabda detektor ustma-ust/yonma-yon turgan 2 ta bankani bitta quti
    deb beradi; keyin suppress_nested aynan haqiqiy bankalarni (kichiklarni) o'chirardi.
    Qoida: quti ichida >= 2 ta kichik quti bo'lsa va ular uning >= cover_thr
    qismini qoplasa — bu guruh. (Shisha ichidagi yorliq + qopqoq bo'laklari odatda
    40% dan kam qoplaydi, shuning uchun ular suppress_nested'ga qoladi.)
    """
    if len(arr) < 3:
        return arr
    xyxy = arr[:, :4]
    areas = np.maximum(0, xyxy[:, 2] - xyxy[:, 0]) * np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
    keep = np.ones(len(arr), dtype=bool)
    for i in np.argsort(-areas):
        if areas[i] <= 0:
            continue
        x1, y1, x2, y2 = xyxy[i]
        ix1 = np.maximum(x1, xyxy[:, 0]); iy1 = np.maximum(y1, xyxy[:, 1])
        ix2 = np.minimum(x2, xyxy[:, 2]); iy2 = np.minimum(y2, xyxy[:, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        inside = (inter / np.maximum(areas, 1e-9) >= contain_thr) & (areas < areas[i]) & keep
        inside[i] = False
        if inside.sum() < 2:
            continue
        mask = np.zeros((grid, grid), dtype=bool)
        sx, sy = grid / max(x2 - x1, 1e-9), grid / max(y2 - y1, 1e-9)
        for j in np.where(inside)[0]:
            a = int(np.clip((ix1[j] - x1) * sx, 0, grid)); b = int(np.ceil(np.clip((ix2[j] - x1) * sx, 0, grid)))
            c = int(np.clip((iy1[j] - y1) * sy, 0, grid)); d = int(np.ceil(np.clip((iy2[j] - y1) * sy, 0, grid)))
            mask[c:d, a:b] = True
        if mask.mean() >= cover_thr:
            keep[i] = False
    return arr[keep]


def is_app_logo(image: np.ndarray, b: Box, corner=(0.22, 0.16), green_min: float = 0.15) -> bool:
    """Agent ilovasining burchakdagi logotipi (yashil "beda bargi") — mahsulot emas.

    Detektorlar uni har bir agent rasmida mahsulot deb topadi. Belgi: chap yuqori
    burchakda, deyarli kvadrat va o'sha yashil rang ko'p (HSV ~ 150°, to'yingan).
    """
    import cv2

    h, w = image.shape[:2]
    if b.x2 > corner[0] * w or b.y2 > corner[1] * h:
        return False
    if not 0.65 <= b.width / max(b.height, 1e-6) <= 1.5:
        return False
    x1, y1, x2, y2 = (max(0, int(v)) for v in b.as_xyxy())
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    green = (hsv[..., 0] >= 60) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 100) & (hsv[..., 2] >= 100)
    return float(green.mean()) >= green_min


class LogoFilter:
    """Har qanday detektor ustidan: ilova logotipi qutilarini olib tashlaydi."""

    def __init__(self, detector):
        self.detector = detector

    def detect(self, image: np.ndarray) -> list[Box]:
        return [b for b in self.detector.detect(image) if not is_app_logo(image, b)]


def nms(boxes: list[Box], iou_thr: float = 0.5) -> list[Box]:
    """Oddiy NMS (ishonch bo'yicha)."""
    if len(boxes) < 2:
        return list(boxes)
    xyxy = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=float)
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort([-b.score for b in boxes])
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(xyxy[i, 0], xyxy[order[1:], 0]); iy1 = np.maximum(xyxy[i, 1], xyxy[order[1:], 1])
        ix2 = np.minimum(xyxy[i, 2], xyxy[order[1:], 2]); iy2 = np.minimum(xyxy[i, 3], xyxy[order[1:], 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return [boxes[i] for i in sorted(keep)]


def _iou_one(b: np.ndarray, many: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(b[0], many[:, 0]); iy1 = np.maximum(b[1], many[:, 1])
    ix2 = np.minimum(b[2], many[:, 2]); iy2 = np.minimum(b[3], many[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    a = (b[2] - b[0]) * (b[3] - b[1])
    am = (many[:, 2] - many[:, 0]) * (many[:, 3] - many[:, 1])
    return inter / (a + am - inter + 1e-9)


def nms_array(arr: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """(N,5) [x1,y1,x2,y2,score] massiv uchun NMS."""
    if len(arr) < 2:
        return arr
    order = np.argsort(-arr[:, 4])
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][_iou_one(arr[i], arr[order[1:]]) < iou_thr]
    return arr[np.sort(keep)]


def tile_windows(w: int, h: int, n: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Rasmni n×n bir-birini `overlap` ulushda qoplaydigan bo'laklarga bo'ladi."""
    def axis(size: int) -> list[tuple[int, int]]:
        t = size / (n - (n - 1) * overlap)
        step = t * (1 - overlap)
        spans = []
        for i in range(n):
            a = int(round(i * step))
            b = size if i == n - 1 else min(size, int(round(i * step + t)))
            spans.append((a, b))
        return spans
    return [(x1, y1, x2, y2) for y1, y2 in axis(h) for x1, x2 in axis(w)]


def cut_by_tile(p: np.ndarray, tw: int, th: int, left: bool, top: bool, right: bool,
                bottom: bool, margin: float = 2.0) -> np.ndarray:
    """Bo'lakning ICHKI chegarasiga tegib turgan (kesilgan) qutilar maskasi.

    Bunday quti mahsulotning bir qismi xolos — u qo'shni bo'lakda yoki butun rasm
    o'tishida to'liq topiladi. Rasmning haqiqiy chetidagi qutilar saqlanadi.
    """
    m = np.zeros(len(p), dtype=bool)
    if left:
        m |= p[:, 0] <= margin
    if top:
        m |= p[:, 1] <= margin
    if right:
        m |= p[:, 2] >= tw - margin
    if bottom:
        m |= p[:, 3] >= th - margin
    return m


def wbf(passes: list[np.ndarray], iou_thr: float = 0.55, min_votes: int = 1) -> np.ndarray:
    """Weighted Box Fusion: bir nechta o'tish qutilarini birlashtiradi.

    NMS eng ishonchli qutini tanlab qolganini tashlaydi; WBF esa bitta mahsulotga
    tushgan barcha qutilarning ishonch bo'yicha o'rtachasini oladi (chegaralar
    aniqroq). Ball = o'rtacha ishonch × (qutini topgan o'tishlar / jami o'tishlar):
    faqat bitta o'tishda chiqqan tasodifiy quti pastroq ball oladi.
    min_votes — quti kamida shuncha o'tishda topilishi shart.
    """
    n_pass = len(passes)
    rows = [np.concatenate([p[:, :5], np.full((len(p), 1), k)], axis=1)
            for k, p in enumerate(passes) if len(p)]
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    allb = np.concatenate(rows)
    allb = allb[np.argsort(-allb[:, 4])]
    fused: list[np.ndarray] = []        # har klaster uchun joriy birlashgan quti
    members: list[list[np.ndarray]] = []
    for b in allb:
        if fused:
            ious = _iou_one(b, np.asarray(fused))
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                members[j].append(b)
                m = np.asarray(members[j])
                wts = m[:, 4:5]
                fused[j] = np.concatenate([(m[:, :4] * wts).sum(0) / wts.sum(), [m[:, 4].mean()]])
                continue
        fused.append(b[:5].copy())
        members.append([b])
    out = []
    for f, m in zip(fused, members):
        votes = len({int(x[5]) for x in m})
        if votes < min_votes:
            continue
        f = f.copy()
        f[4] = f[4] * min(votes, n_pass) / n_pass
        out.append(f)
    return np.asarray(out, dtype=np.float32).reshape(-1, 5)


class EnsembleDetector:
    """Bir nechta detektor natijasini birlashtiradi.

    SKU-110K (zich javon, qutili mahsulotlar) + COCO "bottle/cup" (shaffof PET
    shishalarni BUTUNLIGICHA topadi; SKU-110K ularni ko'pincha yorliq/qopqoq
    bo'laklariga bo'lib yuboradi). Takrorlar NMS bilan, butun shisha ichidagi
    bo'laklar esa suppress_nested bilan olib tashlanadi.
    """

    def __init__(self, detectors: list, iou_thr: float = 0.5, contain_thr: float = 0.7):
        self.detectors = detectors
        self.iou_thr = iou_thr
        self.contain_thr = contain_thr

    def detect(self, image: np.ndarray) -> list[Box]:
        boxes: list[Box] = []
        for d in self.detectors:
            boxes += d.detect(image)
        return suppress_nested(nms(boxes, self.iou_thr), self.contain_thr)


def build_detector(cfg, device: str) -> Detector:
    """config.yaml asosida detektor yaratadi."""
    det = _build_detector(cfg, device)
    if cfg.get("detector.drop_app_logo", True) and not isinstance(det, SimpleDetector):
        det = LogoFilter(det)
    return det


def _build_detector(cfg, device: str) -> Detector:
    backend = cfg.get("detector.backend", "yolo")
    if backend == "simple":
        return SimpleDetector()
    if backend == "zeroshot":
        zs = cfg.get("detector.zeroshot") or {}
        return ZeroShotDetector(zs.get("model", "IDEA-Research/grounding-dino-base"), device,
                                prompts=tuple(zs.get("prompts", ["bottle", "can", "box", "product package"])),
                                conf=zs.get("conf", 0.25), text_conf=zs.get("text_conf", 0.2),
                                iou=zs.get("iou", 0.5), max_area=zs.get("max_area", 0.25))
    tta = cfg.get("detector.tta") or {}
    on = bool(tta.get("enabled", False))
    main = YoloDetector(
        weights=cfg.get("detector.weights", "yolo11m.pt"),
        device=device,
        conf=cfg.get("detector.conf", 0.20),
        iou=cfg.get("detector.iou", 0.55),
        imgsz=cfg.get("detector.imgsz", 1280),
        max_det=cfg.get("detector.max_det", 600),
        agnostic_nms=cfg.get("detector.agnostic_nms", True),
        scales=tta.get("scales") if on else None,
        flip=on and bool(tta.get("flip", False)),
        tiles=tta.get("tiles", 0) if on else 0,
        tile_overlap=tta.get("tile_overlap", 0.25),
        tile_imgsz=tta.get("tile_imgsz"),
        fuse_iou=tta.get("fuse_iou", 0.55),
        min_votes=tta.get("min_votes", 1),
    )
    extra = cfg.get("detector.bottle")
    if backend == "ensemble" and extra and extra.get("weights"):
        from .config import resolve_path
        w = str(resolve_path(extra["weights"]))
        if Path(w).exists():
            bottle = YoloDetector(weights=w, device=device, conf=extra.get("conf", 0.25),
                                  iou=extra.get("iou", 0.55), imgsz=extra.get("imgsz", 1280),
                                  max_det=cfg.get("detector.max_det", 600), agnostic_nms=True,
                                  classes=extra.get("classes", [39, 41]))
            return EnsembleDetector([main, bottle], iou_thr=extra.get("merge_iou", 0.5),
                                    contain_thr=extra.get("contain_thr", 0.7))
    return main
