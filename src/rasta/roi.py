"""ROI (poligon) — rasmda faqat kerakli javon/muzlatgichni tahlil qilish.

Agent rasmida ko'pincha 2–3 ta muzlatgich yoki bir nechta javon bo'ladi, bizga
esa faqat bittasi kerak (masalan kompaniyaning o'z muzlatgichi). Operator uni
poligon bilan belgilaydi (tools/roi_editor.html), tahlil faqat shu zonada o'tadi.

Qo'shimcha foyda — aniqlik: zona kesib olinib kattalashtiriladi (standart:
uzun tomoni 1280 px gacha). 800 px li agent rasmlarida muzlatgich ko'pincha
300–400 px ni egallaydi; kattalashtirilgan zonada detektor mayda mahsulotlarni
ancha yaxshi topadi.

Poligon formati (JSON):
  {"image": "rasm.jpg",
   "regions": [{"name": "Coca-Cola muzlatgich", "points": [[x, y], ...]}, ...]}
Koordinatalar asl rasm pikselida; 0..1 oralig'ida bo'lsa — nisbiy deb olinadi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import AnalysisResult, Box


@dataclass
class Region:
    name: str
    points: np.ndarray                     # (N, 2) float, asl rasm pikselida
    meta: dict = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        x1, y1 = self.points.min(axis=0)
        x2, y2 = self.points.max(axis=0)
        return float(x1), float(y1), float(x2), float(y2)

    def to_dict(self) -> dict:
        return {"name": self.name, "points": [[round(float(x), 1), round(float(y), 1)]
                                              for x, y in self.points]}


def make_region(points, name: str = "zona", image_size: tuple[int, int] | None = None) -> Region:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("Poligon uchun kamida 3 ta nuqta kerak")
    if image_size is not None and pts.max() <= 1.0:      # nisbiy koordinatalar
        pts = pts * np.array(image_size, dtype=float)    # (w, h)
    return Region(name=name, points=pts)


def load_regions(path: str | Path, image_size: tuple[int, int] | None = None) -> list[Region]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    regs = data.get("regions", data if isinstance(data, list) else [])
    return [make_region(r["points"], r.get("name", f"zona{i + 1}"), image_size)
            for i, r in enumerate(regs)]


def parse_points(s: str) -> np.ndarray:
    """"x1,y1;x2,y2;..." -> (N,2)."""
    return np.array([[float(v) for v in p.split(",")] for p in s.split(";") if p.strip()])


def polygon_mask(shape_hw: tuple[int, int], points: np.ndarray) -> np.ndarray:
    import cv2
    m = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillPoly(m, [np.round(points).astype(np.int32)], 1)
    return m.astype(bool)


def inside_fraction(box: Box, mask: np.ndarray) -> float:
    """Box maydonining poligon ichidagi ulushi."""
    h, w = mask.shape
    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = min(w, int(round(box.x2))), min(h, int(round(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].mean())


@dataclass
class RoiWork:
    """Zona bo'yicha ish rasmi va asl koordinatalarga qaytish parametrlari."""
    image: np.ndarray          # kesilgan + kattalashtirilgan (asl piksellar)
    detect_image: np.ndarray   # xuddi shu, poligon tashqarisi kulrang (detektor uchun)
    mask: np.ndarray           # ish rasmidagi poligon maskasi
    scale: float
    ox: float
    oy: float
    Hinv: np.ndarray | None = None   # perspektiva to'g'rilangan bo'lsa: ish rasmi -> asl rasm


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4 nuqtani (yuqori-chap, yuqori-o'ng, past-o'ng, past-chap) tartibiga keltiradi."""
    p = np.asarray(pts, dtype=np.float32)
    by_y = p[np.argsort(p[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bot = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def prepare_rectified(image: np.ndarray, region: Region, target_long: int = 1280,
                      max_scale: float = 3.0) -> RoiWork:
    """4 burchakli zonani homografiya bilan "old tomondan" ko'rinishga keltiradi.

    Operator muzlatgich/javonning 4 burchagini belgilagani uchun homografiya aniq
    hisoblanadi (neyron tarmoq kerak emas). Qatorlar gorizontal, mahsulotlar tik
    bo'ladi — detektor (SKU-110K frontal javonlarda o'qitilgan) va qator mantiqi uchun qulay.
    """
    import cv2
    src = order_corners(region.points)
    wt, wb = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])
    hl, hr = np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])
    ow, oh = max(wt, wb), max(hl, hr)
    if ow < 8 or oh < 8:
        raise ValueError(f"Zona '{region.name}' juda kichik")
    s = float(np.clip(target_long / max(ow, oh), 1.0, max_scale))
    W, H = int(round(ow * s)), int(round(oh * s))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (W, H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    mask = np.ones((H, W), dtype=bool)
    return RoiWork(image=np.ascontiguousarray(warped), detect_image=warped, mask=mask,
                   scale=s, ox=0.0, oy=0.0, Hinv=np.linalg.inv(M))


def prepare(image: np.ndarray, region: Region, target_long: int = 1280,
            max_scale: float = 3.0, pad_frac: float = 0.03, fill: int = 114,
            rectify: bool = False) -> RoiWork:
    import cv2
    if rectify and len(region.points) == 4:
        return prepare_rectified(image, region, target_long, max_scale)
    H, W = image.shape[:2]
    x1, y1, x2, y2 = region.bbox
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1 = int(max(0, np.floor(x1 - pw))), int(max(0, np.floor(y1 - ph)))
    x2, y2 = int(min(W, np.ceil(x2 + pw))), int(min(H, np.ceil(y2 + ph)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f"Zona '{region.name}' juda kichik yoki rasmdan tashqarida")
    crop = image[y1:y2, x1:x2]
    s = float(np.clip(target_long / max(crop.shape[:2]), 1.0, max_scale))
    if s > 1.0:
        crop = cv2.resize(crop, (int(round(crop.shape[1] * s)), int(round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_CUBIC)
    pts = (region.points - np.array([x1, y1])) * s
    mask = polygon_mask(crop.shape[:2], pts)
    det = crop.copy()
    det[~mask] = fill
    return RoiWork(image=np.ascontiguousarray(crop), detect_image=det, mask=mask,
                   scale=s, ox=float(x1), oy=float(y1))


def _map_box(b, s, ox, oy):
    b.x1, b.x2 = b.x1 / s + ox, b.x2 / s + ox
    b.y1, b.y2 = b.y1 / s + oy, b.y2 / s + oy


def to_original(result: AnalysisResult, work: RoiWork, region: Region,
                image_size: tuple[int, int], image_path: str) -> AnalysisResult:
    """Ish rasmidagi koordinatalarni asl rasmga qaytaradi."""
    s, ox, oy = work.scale, work.ox, work.oy
    if work.Hinv is not None:
        # perspektiva: quti burchaklari teskari homografiya bilan -> asl rasmdagi o'rab turuvchi quti
        def warp_back(obj):
            c = np.array([[obj.x1, obj.y1, 1], [obj.x2, obj.y1, 1],
                          [obj.x2, obj.y2, 1], [obj.x1, obj.y2, 1]], dtype=float).T
            q = work.Hinv @ c
            q = q[:2] / q[2]
            obj.x1, obj.y1 = float(q[0].min()), float(q[1].min())
            obj.x2, obj.y2 = float(q[0].max()), float(q[1].max())
        for d in result.detections + result.background:
            warp_back(d.box)
        for g in result.gaps + result.suspects + result.dividers:
            warp_back(g)
            g.width = g.x2 - g.x1
        s = 1.0
    for d in ([] if work.Hinv is not None else result.detections + result.background):
        _map_box(d.box, s, ox, oy)
    for g in ([] if work.Hinv is not None else result.gaps + result.suspects + result.dividers):
        g.x1, g.x2 = g.x1 / s + ox, g.x2 / s + ox
        g.y1, g.y2 = g.y1 / s + oy, g.y2 / s + oy
        g.width = g.width / s
    result.image_path = image_path
    result.image_size = image_size
    result.roi = region
    return result
