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


def _cos(a, b) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def propagate_identical(
    detections: list[Detection],
    sim: float = 0.90,
    max_gap: float = 0.6,
    params: RecognitionParams | None = None,
) -> int:
    """Vizual bir xil qo'shnilar blokiga ishonchli SKU'ni tarqatadi (facing blok).

    Merchandayzingda bir xil mahsulot yonma-yon bloklarda turadi. Agar blokdagi
    kamida bitta a'zo ishonchli tanilgan bo'lsa, embedding o'xshashligi (cos >= sim)
    bo'yicha IDENTIK deb tasdiqlangan qolgan (noma'lum) a'zolar ham shu SKU deb
    qabul qilinadi. `smooth_with_context`dan kuchliroq: ikkala qo'shni ham tanilgan
    bo'lishini talab qilmaydi (blok cheti, bitta tanilgan qo'shni holatlari).

    Faqat NOMA'LUM (yoki brend-darajali emas) a'zolar o'zgartiriladi; tanilgan
    mahsulot hech qachon qayta yozilmaydi. Blokda har xil SKU'lar bo'lsa — o'tkazib
    yuboriladi (noaniq).
    """
    rows: dict[int, list[Detection]] = {}
    for d in detections:
        if d.vec is not None:
            rows.setdefault(d.row, []).append(d)
    fixed = 0
    for row_dets in rows.values():
        row_dets.sort(key=lambda d: d.box.x1)
        n = len(row_dets)
        if n < 2:
            continue
        med_w = float(np.median([d.box.width for d in row_dets]))
        i = 0
        while i < n:
            block = [i]
            j = i
            while j + 1 < n:
                a, b = row_dets[j], row_dets[j + 1]
                if b.box.x1 - a.box.x2 > max_gap * med_w:
                    break
                if _cos(a.vec, b.vec) < sim:
                    break
                block.append(j + 1)
                j += 1
            known_idx = [k for k in block
                         if row_dets[k].match and row_dets[k].match.is_known
                         and row_dets[k].match.level != "brand"]
            skus = {row_dets[k].match.sku_id for k in known_idx}
            if len(block) >= 2 and len(skus) == 1:
                ref = row_dets[known_idx[0]].match
                for k in block:
                    d = row_dets[k]
                    if d.match is None or d.match.is_known or d.match.level == "brand":
                        continue
                    # identiklikni tanilgan a'zo(lar)ga nisbatan tasdiqlash
                    if max(_cos(d.vec, row_dets[m].vec) for m in known_idx) < sim:
                        continue
                    # faqat "yaqin-o'tkazib": item'ning O'Z top-nomzodlari ichida target
                    # bo'lsa (katalog ham shu SKU deb o'ylaydi, faqat chegaradan past) —
                    # bu haqiqiy noma'lumlar va vizual-egizaklarga noto'g'ri tarqalishni to'xtatadi.
                    top2 = (d.match.candidates or [])[:2]
                    cand = next((c for c in top2 if c["sku_id"] == ref.sku_id), None)
                    if cand is None:
                        continue
                    if params is not None:
                        _, min_color, _ = params.limits_for_source(cand.get("src"))
                        if cand.get("color", 1.0) < min_color:
                            continue
                    d.match.sku_id = ref.sku_id
                    d.match.name = ref.name
                    d.match.brand = ref.brand
                    d.match.is_known = True
                    d.match.level = "sku"
                    if cand is not None:
                        d.match.score = float(cand["score"])
                    d.context_fixed = True
                    fixed += 1
            i = j + 1
    return fixed
