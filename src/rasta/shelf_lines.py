"""Javon chetlari (narx relslari) bo'yicha BUTUNLAY bo'sh javon zonalarini topish.

Muammo: oos.py bo'sh joyni mahsulot qatorlari ichidan qidiradi. Agar javonda
umuman mahsulot qolmagan bo'lsa, u "qator" sifatida mavjud emas va hech qachon
OOS deb topilmaydi — holbuki bu eng muhim holat.

Yechim — javon geometriyasini mahsulotlardan emas, rasmning o'zidan olish:
  1. Canny + HoughLinesP -> uzun, deyarli gorizontal segmentlar
     (perspektiva uchun ±25° gacha qiyalik).
  2. Segmentlar bir chiziqqa birlashtiriladi (markazdagi y va burchak bo'yicha)
     -> javon chetlari  y = m·x + b.
  3. Qo'shni chetlar orasidagi zona = javon "qavati".
  4. Himoya (yo'lak, shift, pol chiziqlari uchun): kamida 2 ta mahsulot qatori
     pastki chizig'i biror chetga to'g'ri kelishi kerak (rasm haqiqatan javon).
     Bo'sh zonalar faqat mahsulotli zonalar oralig'ida yoki ulardan ko'pi bilan
     2 qavat pastda qidiriladi, balandligi mahsulotli zonalarga mos bo'lishi kerak.
  5. Zona vertikal tasmalarga bo'linadi; tasma bo'sh = unda turgan mahsulot yo'q
     VA qirralar zichligi mahsulotlarnikidan ancha past. Ketma-ket bo'sh
     tasmalar -> OOS (status "oos", row < 0 = javon zonasi, mahsulot qatori emas).
  6. Ko'p qavatda bir xil x'da takrorlanuvchi tor bo'shliq — bo'linma/ramka.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Detection, Gap


@dataclass
class ShelfEdge:
    m: float            # qiyalik
    b: float            # y = m*x + b
    x1: float
    x2: float
    length: float

    def y(self, x):
        return self.m * np.asarray(x, dtype=float) + self.b


def detect_shelf_edges(image: np.ndarray, min_len_frac: float = 0.15,
                       max_angle: float = 25.0, min_cover_frac: float = 0.3) -> list[ShelfEdge]:
    import cv2

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    L = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=120,
                        minLineLength=int(min_len_frac * W), maxLineGap=int(0.02 * W))
    if L is None:
        return []
    segs = []
    for x1, y1, x2, y2 in L.reshape(-1, 4).astype(float):
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) > max_angle or x2 - x1 < 1:
            continue
        m = (y2 - y1) / (x2 - x1)
        segs.append((y1 + m * (W / 2 - x1), ang, x1, y1, x2, y2, np.hypot(x2 - x1, y2 - y1)))
    segs.sort(key=lambda s: s[0])

    groups: list[list] = []
    for s in segs:
        if groups and abs(s[0] - groups[-1][-1][0]) < 0.015 * H and \
                abs(s[1] - np.mean([g[1] for g in groups[-1]])) < 3.0:
            groups[-1].append(s)
        else:
            groups.append([s])

    out = []
    for g in groups:
        xs = np.array([[s[2], s[4]] for s in g]).ravel()
        ys = np.array([[s[3], s[5]] for s in g]).ravel()
        w = np.repeat([s[6] for s in g], 2)
        x_lo, x_hi = float(xs.min()), float(xs.max())
        if x_hi - x_lo < min_cover_frac * W:
            continue
        A = np.stack([xs, np.ones_like(xs)], 1) * w[:, None]
        m, b = np.linalg.lstsq(A, ys * w, rcond=None)[0]
        out.append(ShelfEdge(float(m), float(b), x_lo, x_hi, float(sum(s[6] for s in g))))
    out.sort(key=lambda e: float(e.y(W / 2)))
    return out


def shelf_support(edges: list[ShelfEdge], detections: list[Detection]) -> float:
    """Mahsulotlarning qancha ulushi javon chetida turadi (pastki chizig'i chetga yaqin).

    Frontal javon rasmlarida ~0.9+, yo'lak/ombor/aralash sahnalarda 0.5–0.7.
    """
    if not detections or not edges:
        return 0.0
    mw = float(np.median([d.box.width for d in detections]))
    ok = 0
    for d in detections:
        ok += any(e.x1 - mw <= d.box.cx <= e.x2 + mw
                  and abs(float(e.y(d.box.cx)) - d.box.y2) < 0.2 * d.box.height for e in edges)
    return ok / len(detections)


def _edge_density_map(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return (cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0).astype(np.float32)


def find_empty_shelves(
    image: np.ndarray,
    detections: list[Detection],
    existing: list[Gap] | None = None,
    strips: int = 48,
    min_rows_explained: int = 2,
    empty_ratio: float = 0.35,
    max_zones_below: int = 2,
    min_support_frac: float = 0.85,
    divider_zone_frac: float = 0.6,
    edges: list[ShelfEdge] | None = None,
) -> tuple[list[Gap], list[ShelfEdge]]:
    """Javon chetlari orasidagi bo'sh zonalarni OOS sifatida qaytaradi."""
    if len(detections) < 4:
        return [], []
    H, W = image.shape[:2]
    if edges is None:
        edges = detect_shelf_edges(image)
    if len(edges) < 3:
        return [], edges

    hs = np.array([d.box.height for d in detections])
    ws = np.array([d.box.width for d in detections])
    med_h, med_w = float(np.median(hs)), float(np.median(ws))

    # Har mahsulot qaysi chet ustida turadi (pastki chiziq chetga yaqin)
    def edge_under(d):
        best, bi = None, -1
        for i, e in enumerate(edges):
            if not (e.x1 - med_w <= d.box.cx <= e.x2 + med_w):
                continue
            dy = abs(float(e.y(d.box.cx)) - d.box.y2)
            if dy < 0.35 * d.box.height and (best is None or dy < best):
                best, bi = dy, i
        return bi

    support = {}
    n_sup = 0
    for d in detections:
        i = edge_under(d)
        if i >= 0:
            support.setdefault(i, []).append(d)
            n_sup += abs(float(edges[i].y(d.box.cx)) - d.box.y2) < 0.2 * d.box.height
    product_edges = sorted(i for i, ds in support.items() if len(ds) >= 2)
    # Rasm haqiqatan frontal javonmi? Toza javon rasmlarida mahsulotlarning ~90%+
    # javon chetida turadi; yo'lak/aralash sahnalarda 50–70% (benchmark: samples/).
    if len(product_edges) < min_rows_explained or n_sup < min_support_frac * len(detections):
        return [], edges

    # Zona = (yuqori chet, pastki chet); mahsulotli zonalarning tipik balandligi
    def zone_height(i):
        j = i - 1
        while j >= 0 and float(edges[i].y(W / 2) - edges[j].y(W / 2)) < 0.5 * med_h:
            j -= 1          # rels qalinligi (yuqori/pastki qirra) — o'tkazib yuborish
        return j, float(edges[i].y(W / 2) - edges[j].y(W / 2)) if j >= 0 else 0.0

    ref_heights = [zone_height(i)[1] for i in product_edges if zone_height(i)[0] >= 0]
    if not ref_heights:
        return [], edges
    ref_h = float(np.median(ref_heights))

    dens = _edge_density_map(image)

    def density(x1, y1, x2, y2):
        x1, x2 = int(max(0, x1)), int(min(W, x2))
        y1, y2 = int(max(0, y1)), int(min(H, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return 0.0
        return float(dens[y1:y2, x1:x2].mean())

    prod_density = float(np.median([density(d.box.x1 + 0.1 * d.box.width, d.box.y1,
                                            d.box.x2 - 0.1 * d.box.width, d.box.y2)
                                    for d in detections]))
    if prod_density <= 1e-6:
        return [], edges

    lo, hi = min(product_edges), max(product_edges)
    gaps: list[Gap] = []
    zones_checked = 0
    seen_lower = set()
    for i in range(1, len(edges)):
        if i < lo or i > hi + max_zones_below * 2 + 1:
            continue
        j, zh = zone_height(i)
        if j < 0 or i in seen_lower or not (0.6 * ref_h <= zh <= 3.0 * ref_h):
            continue
        # Javon oralig'i o'sha joydagi mahsulot bo'yiga mos bo'ladi. Eng yaqin mahsulotli
        # qatordagi mahsulotlardan ancha past "zona" — javon emas, taxta qalinligi yoki
        # taxta bilan orqa reyka orasi (turli o'lchamli mahsulotli javonda global mediana
        # mayda mahsulotlarga tortilib, bunday tirqishni "bo'sh javon" deb ko'rsatardi).
        zone_mid = 0.5 * float(edges[i].y(W / 2) + edges[j].y(W / 2))
        nearest = min(product_edges, key=lambda e: abs(float(edges[e].y(W / 2)) - zone_mid))
        local_h = float(np.median([d.box.height for d in support[nearest]]))
        if zh < 0.6 * local_h:
            continue
        seen_lower.add(i)
        top, bot = edges[j], edges[i]
        zx1, zx2 = max(top.x1, bot.x1), min(top.x2, bot.x2)
        if zx2 - zx1 < 3 * med_w:
            continue
        zones_checked += 1
        step = (zx2 - zx1) / strips
        empty = []
        for k in range(strips):
            sx1, sx2 = zx1 + k * step, zx1 + (k + 1) * step
            xc = 0.5 * (sx1 + sx2)
            y_bot, y_top = float(bot.y(xc)), float(top.y(xc))
            # perspektivada chetlar yaqinlashadi — zona balandligi LOKAL tekshiriladi
            if not (0.6 * ref_h <= y_bot - y_top <= 3.0 * ref_h):
                empty.append(False)
                continue
            occupied = any(d.box.x1 - 0.1 * d.box.width <= xc <= d.box.x2 + 0.1 * d.box.width
                           and d.box.y2 > y_top and d.box.y1 < y_bot for d in detections)
            if occupied:
                empty.append(False)
                continue
            # zona ichki qismi (rels/narx yorlig'i qirralarisiz)
            ya, yb = y_top + 0.15 * (y_bot - y_top), y_bot - 0.1 * (y_bot - y_top)
            empty.append(density(sx1, ya, sx2, yb) < empty_ratio * prod_density)
        # ketma-ket bo'sh tasmalar -> bo'shliq
        k = 0
        while k < strips:
            if not empty[k]:
                k += 1
                continue
            s = k
            while k < strips and empty[k]:
                k += 1
            gx1, gx2 = zx1 + s * step, zx1 + k * step
            if gx2 - gx1 < 1.2 * med_w:
                continue
            ys = [float(top.y(gx1)), float(top.y(gx2)), float(bot.y(gx1)), float(bot.y(gx2))]
            if max(ys[2:]) - min(ys[:2]) <= 0:
                continue
            g = Gap(row=-(i + 1), x1=gx1, y1=min(ys[:2]), x2=gx2, y2=max(ys[2:]),
                    width=gx2 - gx1, est_missing_facings=max(1, int(round((gx2 - gx1) / med_w))),
                    emptiness=1.0, status="oos")
            gaps.append(g)

    # Bo'linma/ramka filtri: tor bo'shliq ko'p qavatlarda bir xil x'da -> tuzilma
    def xov(a, b):
        ov = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        return ov / max(1e-6, min(a.width, b.width))
    if zones_checked >= 3:
        keep = []
        for g in gaps:
            zones = {o.row for o in gaps if xov(g, o) >= 0.5}
            if g.width < 2.5 * med_w and len(zones) >= max(3, divider_zone_frac * zones_checked):
                continue
            keep.append(g)
        gaps = keep

    # mavjud (qator ichidagi) OOS bilan ustma-ust tushganlarni olib tashlash
    def overlap(a, b):
        ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        return ix * iy / max(1e-6, (a.x2 - a.x1) * (a.y2 - a.y1))
    existing = existing or []
    gaps = [g for g in gaps if all(overlap(g, e) < 0.3 for e in existing)]
    # Dublikatlar (bitta javonning bir nechta qirrasi bir xil zonani beradi) — NMS
    gaps.sort(key=lambda g: -(g.x2 - g.x1) * (g.y2 - g.y1))
    kept: list[Gap] = []
    for g in gaps:
        if all(overlap(g, k) < 0.5 for k in kept):
            kept.append(g)
    return kept, edges
