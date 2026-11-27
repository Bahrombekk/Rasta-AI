"""Javon konteksti — qo'shni mahsulotlar bo'yicha tanishni tuzatish.

Merchandayzingda bir xil SKU odatda yonma-yon bloklarda turadi (facing blok).
Agar mahsulot noma'lum chiqsa (yorug'lik, blik, qisman to'silish), lekin chap
va o'ng qo'shnisi bir xil SKU bo'lsa VA shu SKU uning o'z nomzodlari orasida
chegaraga yaqin ball bilan bo'lsa — u shu SKU deb qabul qilinadi.

Ehtiyotkor qoida: faqat NOMA'LUM natijalar tuzatiladi; tanilgan mahsulot hech
qachon qo'shnilar tufayli o'zgartirilmaydi. Rang veto'si saqlanadi.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .catalog import RecognitionParams
from .features import shape_similarity
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
            # brend/variant darajasidagilar qo'shniga qarab SKU'ga aylantirilmaydi
            # (ikki kivi orasidagi nok — kivi emas)
            if d.match is None or d.match.is_known or d.match.level or not d.match.candidates:
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
            d.match.family = ref.family
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


def recover_occluded_family(detections, colors: np.ndarray, params: RecognitionParams) -> int:
    """Use a confident, overlapping facing to confirm a borderline family match."""
    anchors: dict[tuple[int, str], list[int]] = {}
    for i, d in enumerate(detections):
        m = d.match
        if (m and m.family and m.is_known and m.sku_id and -1 < d.row
                and not d.context_fixed and d.vec is not None and m.candidates):
            threshold = params.limits_for_source(m.candidates[0].get("src"))[0]
            if m.score >= threshold + 0.04:
                anchors.setdefault((d.row, m.sku_id), []).append(i)

    recovered = 0
    for i, d in enumerate(detections):
        m = d.match
        if m is None or m.is_known or m.level is not None or d.vec is None or not m.candidates:
            continue
        top = m.candidates[0]
        nearby = anchors.get((d.row, top["sku_id"]), ())
        if not nearby:
            continue
        tau, color_min, neg_margin = params.limits_for_source(top.get("src"))
        embed_min = params.embed_min_for(top.get("src"))
        if not tau - 0.03 <= m.score < tau or m.margin < params.min_margin:
            continue
        if top["color"] < color_min or (embed_min is not None and top["dino"] < embed_min):
            continue
        if m.negative_score is not None and neg_margin is not None and m.score < m.negative_score - neg_margin:
            continue
        for j in nearby:
            ref = detections[j]
            a, b = d.box, ref.box
            overlap_x = min(a.x2, b.x2) - max(a.x1, b.x1)
            overlap_y = min(a.y2, b.y2) - max(a.y1, b.y1)
            if not (a.width <= 0.9 * b.width and 0.65 * b.height <= a.height <= 1.15 * b.height
                    and 0.15 * a.width <= overlap_x <= 0.55 * a.width
                    and overlap_y >= 0.8 * a.height and abs(a.y2 - b.y2) <= 0.2 * b.height):
                continue
            if _cos(d.vec, ref.vec) < 0.78 or float(colors[i] @ colors[j]) < 0.90:
                continue
            d.match = replace(m, sku_id=ref.match.sku_id, name=ref.match.name, brand=ref.match.brand,
                              is_known=True, level="sku", family=True)
            d.context_fixed = True
            recovered += 1
            break
    return recovered


def recover_cluster_brands(detections, vectors, colors, aspects, cfg, params: RecognitionParams) -> int:
    """Confirm a brand from a same-row cluster with several independently known examples."""
    from .cluster import cluster_unknown_features

    def brand_key(value):
        return " ".join((value or "").casefold().split())

    rows_with_anchors = {
        d.row for d in detections if d.row >= 0 and d.match and d.match.brand
        and (d.match.is_known or d.match.level == "brand") and not d.context_fixed
    }
    pending = []
    for i, d in enumerate(detections):
        m = d.match
        if d.row not in rows_with_anchors or not m or m.is_known or m.level or m.mismatch or not m.candidates:
            continue
        top = m.candidates[0]
        brand = brand_key(top.get("brand"))
        tau, color_min, neg_margin = params.limits_for_source(top.get("src"))
        if not brand or m.score < tau - 0.07 or top.get("color", 0.0) < max(0.75, color_min):
            continue
        rival = next((c for c in m.candidates[1:] if brand_key(c.get("brand")) != brand), None)
        if rival and m.score - rival["score"] < max(0.04, params.brand_rival_margin or 0.0):
            continue
        pending.append((i, brand, tau, neg_margin))
    if not pending:
        return 0

    labels = cluster_unknown_features(vectors, colors, np.asarray(aspects), cfg)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(detections):
        if labels[i] >= 0:
            groups.setdefault((d.row, int(labels[i])), []).append(i)
    recovered = 0
    for i, brand, tau, neg_margin in pending:
        d = detections[i]
        members = groups.get((d.row, int(labels[i])), ())
        confirmed = [j for j in members if detections[j].match and detections[j].match.brand
                     and (detections[j].match.is_known or detections[j].match.level == "brand")
                     and not detections[j].context_fixed]
        if len(confirmed) < 3 or 2 * len(confirmed) <= len(members):
            continue
        if any(brand_key(detections[j].match.brand) != brand for j in confirmed):
            continue
        support = []
        for j in confirmed:
            visual = _cos(vectors[i], vectors[j])
            color = float(colors[i] @ colors[j])
            if visual < 0.54 or color < 0.77:
                continue
            shape = shape_similarity(aspects[i], aspects[j], params.shape_sigma)
            score = (1 - params.w_color - params.w_shape) * visual + params.w_color * color + params.w_shape * shape
            if score >= tau - 0.09:
                support.append(score)
        if len(support) < 2:
            continue
        negative = d.match.negative_score
        if (negative is not None and neg_margin is not None
                and max(support) < negative - min(0.07, neg_margin + 0.04)):
            continue
        d.match = replace(d.match, brand=detections[confirmed[0]].match.brand, level="brand")
        d.context_fixed = True
        recovered += 1
    return recovered


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
                    if d.match is None or d.match.is_known or d.match.level:
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
                    d.match.family = ref.family
                    d.match.is_known = True
                    d.match.level = "sku"
                    if cand is not None:
                        d.match.score = float(cand["score"])
                    d.context_fixed = True
                    fixed += 1
            i = j + 1
    return fixed


def veto_size_outliers(
    detections: list[Detection],
    params: RecognitionParams | None = None,
    sim_threshold: float = 0.0,
    low: float = 0.6,
    min_support: int = 3,
    strong: float = 0.08,
) -> int:
    """Bitta rasmda bir SKU'ning barcha facing'lari deyarli bir xil bo'yda bo'ladi (bir xil
    mahsulot, bir xil masofa). Ko'pchilikdan keskin kichik/katta VA chegaraga yaqin (kuchsiz)
    mos kelgan quti — o'xshash rangli boshqa mahsulot: u noma'lumga qaytariladi.

    Real misol: 1 litrli sok kartoni katalogi bilan undan ikki baravar kichik sariq bolalar
    sharbati 0.65–0.67 o'xshashlik bilan "tanilgan". Kuchli mos kelganlarga (chegaradan
    `strong` yuqori) tegilmaydi — qiya kadrda masshtab haqiqatan har xil bo'lishi mumkin.
    Qaytaradi: bekor qilingan tanishlar soni.
    """
    groups: dict[str, list[Detection]] = {}
    for d in detections:
        m = d.match
        if m and m.is_known and m.sku_id and m.level != "brand":
            groups.setdefault(m.sku_id, []).append(d)
    vetoed = 0
    for members in groups.values():
        if len(members) <= min_support:
            continue
        typical = float(np.median([d.box.height for d in members]))
        for d in members:
            ratio = d.box.height / max(typical, 1e-6)
            if low <= ratio <= 1.0 / low:
                continue
            m = d.match
            tau = sim_threshold
            if params is not None and m.candidates:
                tau = params.limits_for_source(m.candidates[0].get("src"))[0]
            if m.score >= tau + strong:
                continue
            m.sku_id = m.name = m.brand = m.level = None
            m.is_known = False
            d.context_fixed = False
            vetoed += 1
    return vetoed
