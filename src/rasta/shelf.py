"""Javon qatorlarini aniqlash — bbox'larni gorizontal qatorlarga klasterlash.

Asosiy g'oya: mahsulot javon taxtasiga PASTKI cheti bilan turadi. Shuning uchun
bir qatordagi baland shisha va past bankaning y-markazlari har xil, lekin
pastki chiziqlari (y2) deyarli bir xil. Klasterlash "anchor" bo'yicha:
  anchor = "bottom" (standart) -> y2, "center" -> y-markaz (osma mahsulotlar uchun).

Algoritm: anchor bo'yicha saralanadi; har yangi box joriy qatorning o'rtacha
anchor qiymatidan `row_tolerance * median_balandlik` dan uzoq bo'lsa — yangi
qator boshlanadi (avvalgi qo'shni bilan emas, qator o'rtachasi bilan solishtirish
"zanjir siljishi"ning oldini oladi).

Qo'shimcha: "orqa qator" filtri. Yuqori javondagi mahsulotlar ortida qisman
ko'rinadigan shisha bo'yinlari alohida, past bo'yli "qator" hosil qiladi. Bunday
qator o'rtacha balandligi qo'shni qatordan ancha kichik va u bilan vertikal
ustma-ust tushsa — fon deb belgilanadi (KPI va OOS'ga kirmaydi).
"""
from __future__ import annotations

import numpy as np

from .types import Detection


def _anchor(d: Detection, anchor: str) -> float:
    return d.box.y2 if anchor == "bottom" else d.box.cy


def assign_rows(
    detections: list[Detection],
    row_tolerance: float = 0.6,
    anchor: str = "bottom",
) -> int:
    """Detection'larni javon qatorlariga ajratadi (0 = eng tepa). Qatorlar sonini qaytaradi."""
    if not detections:
        return 0

    heights = np.array([d.box.height for d in detections], dtype=float)
    med_h = float(np.median(heights)) if len(heights) else 1.0
    tol = max(1.0, med_h * row_tolerance)

    order = sorted(range(len(detections)), key=lambda i: _anchor(detections[i], anchor))
    row_idx = 0
    row_vals: list[float] = []
    for i in order:
        a = _anchor(detections[i], anchor)
        if row_vals and abs(a - float(np.mean(row_vals))) > tol:
            row_idx += 1
            row_vals = []
        detections[i].row = row_idx
        row_vals.append(a)
    return row_idx + 1


def _band(dets: list[Detection]) -> tuple[float, float, float]:
    y1 = float(np.median([d.box.y1 for d in dets]))
    y2 = float(np.median([d.box.y2 for d in dets]))
    return y1, y2, y2 - y1


def mark_background_rows(
    detections: list[Detection],
    num_rows: int,
    height_ratio: float = 0.6,
    overlap_frac: float = 0.25,
) -> set[int]:
    """Orqa fon / qisman ko'rinadigan qatorlarni topadi va ularni detections'dan
    ajratish uchun qator indekslarini qaytaradi."""
    rows = {r: [d for d in detections if d.row == r] for r in range(num_rows)}
    bands = {r: _band(ds) for r, ds in rows.items() if ds}
    bg: set[int] = set()
    for r, (y1, y2, h) in bands.items():
        for s, (sy1, sy2, sh) in bands.items():
            if s == r or h >= height_ratio * sh:
                continue
            # Fon qatori kattaroq qatordan ancha siyrak bo'ladi (bir nechta bo'yin);
            # kichik mahsulotli haqiqiy qatorda esa mahsulotlar odatda ko'proq.
            if len(rows[r]) >= 0.5 * len(rows[s]):
                continue
            ov = max(0.0, min(y2, sy2) - max(y1, sy1))
            # past bo'yli qator kattaroq qator bilan ustma-ust tushadi yoki
            # uning tepasiga "yopishgan" (bo'yinlar yorliq tepasida tugaydi)
            touching = abs(sy1 - y2) <= 0.15 * sh
            if ov >= overlap_frac * h or touching:
                bg.add(r)
                break
    return bg


def mark_cut_rows(
    detections: list[Detection],
    num_rows: int,
    image_height: int,
    edge_frac: float = 0.006,
    touch_frac: float = 0.6,
    height_ratio: float = 0.5,
) -> set[int]:
    """Kadr cheti kesib ketgan qatorlar: yuqoridagi javon mahsulotlarining pastki bo'lagi
    (yoki pastdagisining tepasi). Belgisi — qutilarining ko'pi rasm chetiga tegadi VA ular
    qo'shni qatordagidan ancha past (to'liq ko'ringan qator esa chetga tegsa ham o'z
    bo'yida bo'ladi). Bunday bo'laklar tanilmaydi, sanalmaydi va bo'sh joy bermaydi."""
    rows = {r: [d for d in detections if d.row == r] for r in range(num_rows)}
    cut: set[int] = set()
    for r, ds in rows.items():
        if not ds:
            continue
        top = float(np.mean([d.box.y1 <= edge_frac * image_height for d in ds]))
        bottom = float(np.mean([d.box.y2 >= (1 - edge_frac) * image_height for d in ds]))
        if max(top, bottom) < touch_frac:
            continue
        neighbour = rows.get(r + 1 if top >= bottom else r - 1)
        if neighbour and _band(ds)[2] < height_ratio * _band(neighbour)[2]:
            cut.add(r)
    return cut


def renumber_rows(detections: list[Detection]) -> int:
    """Qator indekslarini 0..N-1 qilib qayta raqamlaydi (yuqoridan pastga)."""
    rows = sorted({d.row for d in detections})
    remap = {r: i for i, r in enumerate(rows)}
    for d in detections:
        d.row = remap[d.row]
    return len(rows)


def row_bounds(detections: list[Detection], row: int) -> tuple[float, float, float, float]:
    """Berilgan qatordagi barcha bbox'larni qamrab oluvchi chegara (x1,y1,x2,y2)."""
    xs1, ys1, xs2, ys2 = [], [], [], []
    for d in detections:
        if d.row == row:
            xs1.append(d.box.x1); ys1.append(d.box.y1)
            xs2.append(d.box.x2); ys2.append(d.box.y2)
    if not xs1:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs1), min(ys1), max(xs2), max(ys2)


def row_band(detections: list[Detection], row: int) -> tuple[float, float]:
    """Qatorning "tipik" vertikal oralig'i (mediana y1..y2) — OOS qutisi uchun."""
    ds = [d for d in detections if d.row == row]
    if not ds:
        return 0.0, 0.0
    y1, y2, _ = _band(ds)
    return y1, y2
