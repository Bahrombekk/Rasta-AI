"""OOS (Out-of-Stock) — bo'sh joy aniqlash algoritmi (hisobotning 13-bo'limi).

Bo'sh joy obyekt emas: uning chegarasi qo'shnilarga bog'liq. Shuning uchun
detektor klassi bilan emas, geometriya bilan aniqlanadi:
  1. Har qator uchun band segmentlar (bbox'lar birlashmasi) hisoblanadi.
  2. Bo'sh segmentlar = qator kengligi − band segmentlar.
  3. Qo'shni mahsulot o'rtacha kengligidan katta bo'sh segment => OOS nomzod.

Yolg'on pozitivlarni kamaytirish: javonning vertikal bo'linmasi (yog'och
ustun, seksiyalar orasidagi bo'shliq) barcha qatorlarda bir xil x-pozitsiyada
paydo bo'ladi. Haqiqiy OOS odatda bitta qatorda bo'ladi. Shuning uchun
qatorlar bo'ylab vertikal moslashgan bo'shliqlar tuzilma deb hisoblanib,
OOS'dan chiqarib tashlanadi (13-bo'limdagi "prior + qoida" g'oyasi).

Rasm bo'yicha tekshiruv (ixtiyoriy, image berilganda): geometrik bo'shliq ichida
detektor o'tkazib yuborgan mahsulot bo'lishi mumkin. Bo'sh javon odatda tekis
(kam qirra, kam rang xilma-xilligi), mahsulot esa — yorliq, matn, logotip.
Bo'shliqdagi qirralar zichligi shu qatordagi mahsulotlarnikiga yaqin bo'lsa,
u "suspect" (shubhali) deb belgilanadi va OOS hisobiga kirmaydi.
"""
from __future__ import annotations

import numpy as np

from .types import Detection, Gap
from .shelf import row_band, row_bounds


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Ustma-ust tushgan [x1,x2] oraliqlarni birlashtiradi."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _overlap_ratio(a: Gap, b: Gap) -> float:
    """Ikki bo'shliqning x bo'yicha ustma-ustligi (kichigiga nisbatan)."""
    ov = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    denom = min(a.width, b.width)
    return ov / denom if denom > 0 else 0.0


def _filter_dividers(
    gaps: list[Gap],
    num_rows: int,
    divider_row_frac: float,
    align_ratio: float = 0.35,
) -> tuple[list[Gap], list[Gap]]:
    """Qatorlar bo'ylab vertikal moslashgan bo'shliqlarni (bo'linma) ajratadi.

    Qaytaradi: (haqiqiy_oos, bo'linmalar).
    """
    if num_rows < 3 or len(gaps) < 2:
        return gaps, []  # kam qatorda bo'linmani ishonchli aniqlab bo'lmaydi

    need_rows = max(3, int(np.ceil(num_rows * divider_row_frac)))

    real: list[Gap] = []
    dividers: list[Gap] = []
    for g in gaps:
        rows_hit = {g.row}
        for other in gaps:
            if other is g:
                continue
            if _overlap_ratio(g, other) >= align_ratio:
                rows_hit.add(other.row)
        if len(rows_hit) >= need_rows:
            dividers.append(g)
        else:
            real.append(g)
    return real, dividers


def _edge_density(gray_edges: np.ndarray, x1, y1, x2, y2) -> float:
    h, w = gray_edges.shape[:2]
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    return float(gray_edges[y1:y2, x1:x2].mean())


def verify_gaps(
    image: np.ndarray,
    gaps: list[Gap],
    detections: list[Detection],
    suspect_ratio: float = 0.55,
) -> tuple[list[Gap], list[Gap]]:
    """Bo'shliqlarni rasm bo'yicha tekshiradi. Qaytaradi: (oos, suspects)."""
    import cv2

    if not gaps:
        return [], []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = (cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0).astype(np.float32)

    ref: dict[int, float] = {}
    for g in gaps:
        if g.row not in ref:
            dens = [_edge_density(edges, d.box.x1 + 0.1 * d.box.width, d.box.y1,
                                  d.box.x2 - 0.1 * d.box.width, d.box.y2)
                    for d in detections if d.row == g.row]
            ref[g.row] = float(np.median(dens)) if dens else 0.0

    oos, suspects = [], []
    for g in gaps:
        # chekkalardagi qo'shni mahsulot qirralarini hisobga olmaslik uchun ichki qism
        pad = 0.12 * g.width
        dens = _edge_density(edges, g.x1 + pad, g.y1, g.x2 - pad, g.y2)
        r = ref.get(g.row, 0.0)
        rel = dens / r if r > 1e-6 else 0.0
        g.emptiness = float(np.clip(1.0 - rel, 0.0, 1.0))
        if r > 1e-6 and rel >= suspect_ratio:
            g.status = "suspect"
            suspects.append(g)
        else:
            oos.append(g)
    return oos, suspects


def detect_gaps(
    detections: list[Detection],
    num_rows: int,
    min_gap_ratio: float = 0.8,
    edge_margin: float = 0.02,
    divider_row_frac: float = 0.6,
    return_dividers: bool = False,
    use_shelf_extent: bool = True,
):
    """Har bir javon qatoridagi bo'sh joylarni topadi (bo'linmalarni chiqarib).

    use_shelf_extent=True: qator chegarasi faqat o'z mahsulotlari bilan emas,
    butun javon kengligi bilan olinadi — shunda qator OXIRI yoki BOSHIDAGI
    bo'sh kataklar ham topiladi (aks holda ular hech qachon ko'rinmaydi).
    Javon kengligi = qatorlar x1 ning 25-persentili .. x2 ning 75-persentili
    (bitta uzun/qisqa qator natijani buzmasligi uchun).

    return_dividers=True bo'lsa (real_oos, dividers) juftligini qaytaradi.
    """
    candidates: list[Gap] = []

    shelf_x1 = shelf_x2 = None
    if use_shelf_extent:
        ext = []
        for row in range(num_rows):
            rd = [d for d in detections if d.row == row]
            if len(rd) >= 3:
                ext.append((min(d.box.x1 for d in rd), max(d.box.x2 for d in rd)))
        if len(ext) >= 2:
            shelf_x1 = float(np.percentile([e[0] for e in ext], 25))
            shelf_x2 = float(np.percentile([e[1] for e in ext], 75))

    for row in range(num_rows):
        row_dets = [d for d in detections if d.row == row]
        if len(row_dets) < 2:
            continue  # bitta mahsulotdan bo'sh joyni ishonchli o'lchab bo'lmaydi

        rx1, _, rx2, _ = row_bounds(detections, row)
        if shelf_x1 is not None:
            rx1, rx2 = min(rx1, shelf_x1), max(rx2, shelf_x2)
        ry1, ry2 = row_band(detections, row)
        row_width = rx2 - rx1
        if row_width <= 0:
            continue

        widths = np.array([d.box.width for d in row_dets], dtype=float)
        avg_w = float(np.median(widths))
        min_gap = avg_w * min_gap_ratio
        margin = row_width * edge_margin

        occupied = _merge_intervals([(d.box.x1, d.box.x2) for d in row_dets])

        # Band segmentlar orasidagi bo'shliqlar (chekka bo'shliqlarni hisobga olmaymiz)
        cursor = rx1 + margin
        limit = rx2 - margin
        for seg_a, seg_b in occupied:
            if seg_a - cursor >= min_gap:
                candidates.append(_make_gap(row, cursor, seg_a, ry1, ry2, avg_w))
            cursor = max(cursor, seg_b)
        # oxirgi band segmentdan qator oxirigacha
        if limit - cursor >= min_gap:
            candidates.append(_make_gap(row, cursor, limit, ry1, ry2, avg_w))

    real, dividers = _filter_dividers(candidates, num_rows, divider_row_frac)
    for g in dividers:
        g.status = "divider"

    if return_dividers:
        return real, dividers
    return real


def _make_gap(row, x1, x2, y1, y2, avg_w) -> Gap:
    width = x2 - x1
    est = max(1, int(round(width / avg_w))) if avg_w > 0 else 1
    return Gap(row=row, x1=x1, y1=y1, x2=x2, y2=y2, width=width,
               est_missing_facings=est)
