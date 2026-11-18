"""Biznes KPI hisoblash (hisobotning 2-bo'limi).

Muhim prinsip: raqamlarni CV qatlami sanaydi, VLM emas. Bu yerda faqat
detektsiya+tanish natijalaridan aniq metrikalar hisoblanadi.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from .types import Detection, Gap


def compute_kpi(
    detections: list[Detection],
    gaps: list[Gap],
    num_rows: int,
    brand_of: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Share-of-Shelf, facings, OOS va tanish sifat metrikalarini hisoblaydi.

    brand_of: sku_id -> brend nomi (raqobatchi ulushi uchun, ixtiyoriy).
    """
    total = len(detections)
    known = [d for d in detections if d.match and d.match.is_known]
    unknown = total - len(known)

    total_area = sum(d.box.area for d in detections) or 1.0

    facings: dict[str, int] = defaultdict(int)
    area_by_sku: dict[str, float] = defaultdict(float)
    name_by_sku: dict[str, str] = {}
    for d in known:
        sku = d.match.sku_id
        facings[sku] += 1
        area_by_sku[sku] += d.box.area
        name_by_sku[sku] = d.match.name or sku

    # Share of Shelf (maydon bo'yicha) — jami detektsiya maydoniga nisbatan
    share_of_shelf = {
        sku: {
            "name": name_by_sku[sku],
            "facings": facings[sku],
            "share_pct": round(100.0 * area_by_sku[sku] / total_area, 2),
        }
        for sku in facings
    }
    share_of_shelf = dict(
        sorted(share_of_shelf.items(), key=lambda kv: kv[1]["share_pct"], reverse=True)
    )

    # Brend darajasidagi ulush (raqobatchi tahlili)
    brand_share: dict[str, dict] = {}
    # SKU noaniq, lekin brend aniq bo'lgan mahsulotlar ham brend ulushiga kiradi
    brand_only = [d for d in detections if d.match and not d.match.is_known
                  and d.match.level == "brand" and d.match.brand]
    if brand_of or brand_only:
        brand_of = brand_of or {}
        b_facings: dict[str, int] = defaultdict(int)
        b_area: dict[str, float] = defaultdict(float)
        for d in known:
            brand = brand_of.get(d.match.sku_id) or d.match.brand or "?"
            b_facings[brand] += 1
            b_area[brand] += d.box.area
        for d in brand_only:
            b_facings[d.match.brand] += 1
            b_area[d.match.brand] += d.box.area
        brand_share = {
            b: {
                "facings": b_facings[b],
                "share_pct": round(100.0 * b_area[b] / total_area, 2),
            }
            for b in sorted(b_facings, key=lambda x: b_area[x], reverse=True)
        }

    # OOS xulosasi
    oos_by_row: dict[int, int] = defaultdict(int)
    shelf_zone = [g for g in gaps if g.row < 0]      # butunlay bo'sh javon zonalari
    for g in gaps:
        if g.row >= 0:
            oos_by_row[g.row] += g.est_missing_facings

    return {
        "num_products": total,
        "num_known": len(known),
        "num_unknown": unknown,
        "num_brand_only": len(brand_only),
        "recognition_rate_pct": round(100.0 * len(known) / total, 1) if total else 0.0,
        "num_rows": num_rows,
        "distinct_skus": len(facings),
        "facings": dict(facings),
        "share_of_shelf": share_of_shelf,
        "brand_share": brand_share,
        "oos": {
            "num_gaps": len(gaps),
            "est_missing_facings": sum(g.est_missing_facings for g in gaps),
            "by_row": {str(k): v for k, v in sorted(oos_by_row.items())},
            "empty_shelf_zones": len(shelf_zone),
            "empty_shelf_facings": sum(g.est_missing_facings for g in shelf_zone),
        },
    }
