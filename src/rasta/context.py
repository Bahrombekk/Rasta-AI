"""Javon konteksti — qo'shni mahsulotlar bo'yicha tanishni tuzatish.

Merchandayzingda bir xil SKU odatda yonma-yon bloklarda turadi (facing blok).
Agar mahsulot noma'lum chiqsa (yorug'lik, blik, qisman to'silish), lekin chap
va o'ng qo'shnisi bir xil SKU bo'lsa VA shu SKU uning o'z nomzodlari orasida
chegaraga yaqin ball bilan bo'lsa — u shu SKU deb qabul qilinadi.

Ehtiyotkor qoida: faqat NOMA'LUM natijalar tuzatiladi; tanilgan mahsulot hech
qachon qo'shnilar tufayli o'zgartirilmaydi. Rang veto'si saqlanadi.
"""
from __future__ import annotations

import numpy as np

from .catalog import RecognitionParams
from .types import Detection


def _neighbors(row_dets: list[Detection], idx: int, max_gap: float):
    d = row_dets[idx]
    left = right = None
    if idx > 0:
        L = row_dets[idx - 1]
        if d.box.x1 - L.box.x2 <= max_gap:
            left = L
    if idx + 1 < len(row_dets):
        R = row_dets[idx + 1]
        if R.box.x1 - d.box.x2 <= max_gap:
            right = R
    return left, right


def smooth_with_context(
    detections: list[Detection],
    sim_threshold: float,
    relax: float = 0.08,
    color_min: float = 0.0,
    params: RecognitionParams | None = None,
) -> int:
    """Noma'lum detection'larni qo'shnilar bo'yicha tuzatadi. Tuzatilganlar sonini qaytaradi."""
    rows: dict[int, list[Detection]] = {}
    for d in detections:
        rows.setdefault(d.row, []).append(d)

    fixed = 0
    for row_dets in rows.values():
        row_dets.sort(key=lambda d: d.box.x1)
        if len(row_dets) < 3:
            continue
        med_w = float(np.median([d.box.width for d in row_dets]))
        updates = []
        for i, d in enumerate(row_dets):
            if d.match is None or d.match.is_known or d.match.level == "brand" or not d.match.candidates:
                continue
            left, right = _neighbors(row_dets, i, max_gap=0.5 * med_w)
            if left is None or right is None:
                continue
            if not (left.match and right.match and left.match.is_known
                    and right.match.is_known and left.match.sku_id == right.match.sku_id):
                continue
            target = left.match.sku_id
            cand = next((c for c in d.match.candidates if c["sku_id"] == target), None)
            if cand is None:
                continue
            if params is None:
                tau, min_color, neg_margin = sim_threshold, color_min, None
            else:
                tau, min_color, neg_margin = params.limits_for_source(cand.get("src"))
            neg = d.match.negative_score
            if neg is not None and neg_margin is not None and cand["score"] < neg - neg_margin:
                continue
            if cand["score"] >= tau - relax and cand.get("color", 1.0) >= min_color:
                updates.append((d, target, cand, left.match))
        for d, target, cand, ref in updates:
            d.match.sku_id = target
            d.match.name = ref.name
            d.match.brand = ref.brand
            d.match.is_known = True
            d.match.level = "sku"
            d.match.score = float(cand["score"])
            d.context_fixed = True
            fixed += 1
    return fixed
