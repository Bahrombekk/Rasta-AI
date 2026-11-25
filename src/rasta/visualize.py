"""Tahlil natijasini rasm ustiga chizish (annotatsiya)."""
from __future__ import annotations

import colorsys
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import AnalysisResult


def _color_for(key: str) -> tuple[int, int, int]:
    """SKU/ID uchun barqaror rang (crc32 — har ishga tushishda bir xil)."""
    h = (zlib.crc32(key.encode("utf-8")) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int = 14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _dashed_rect(draw, xy, color, width=2, dash=8):
    x1, y1, x2, y2 = xy
    for x in range(int(x1), int(x2), dash * 2):
        draw.line([x, y1, min(x + dash, x2), y1], fill=color, width=width)
        draw.line([x, y2, min(x + dash, x2), y2], fill=color, width=width)
    for y in range(int(y1), int(y2), dash * 2):
        draw.line([x1, y, x1, min(y + dash, y2)], fill=color, width=width)
        draw.line([x2, y, x2, min(y + dash, y2)], fill=color, width=width)


def dim_outside(image: np.ndarray, regions, factor: float = 0.35) -> np.ndarray:
    """Zonalar (poligonlar) tashqarisini xiralashtiradi."""
    from .roi import polygon_mask
    m = np.zeros(image.shape[:2], dtype=bool)
    for r in regions:
        m |= polygon_mask(image.shape[:2], r.points)
    out = image.copy()
    out[~m] = (out[~m] * factor + 255 * (1 - factor) * 0.15).astype(np.uint8)
    return out


def _draw_roi(draw, region, font, lw):
    pts = [tuple(map(float, p)) for p in region.points]
    draw.line(pts + [pts[0]], fill=(0, 220, 120, 255), width=lw + 1)
    x, y = min(p[0] for p in pts), min(p[1] for p in pts)
    tb = draw.textbbox((0, 0), region.name, font=font)
    draw.rectangle([x, max(0, y - (tb[3] - tb[1]) - 6), x + tb[2] - tb[0] + 8, max(0, y)],
                   fill=(0, 220, 120, 230))
    draw.text((x + 4, max(0, y - (tb[3] - tb[1]) - 4)), region.name, fill=(0, 0, 0), font=font)


def draw_results(image: np.ndarray, results: list, show_scores: bool = True) -> np.ndarray:
    """Bir rasmdagi bir nechta zona natijasini bitta rasmda chizadi."""
    regs = [r.roi for r in results if r.roi is not None]
    canvas = dim_outside(image, regs) if regs else image
    for r in results:
        canvas = draw_result(canvas, r, show_scores=show_scores, dim=False)
    return canvas


def draw_result(image: np.ndarray, result: AnalysisResult, show_scores: bool = True,
                dim: bool = True) -> np.ndarray:
    """Detektsiya qutilari, SKU nomlari, noma'lum klasterlar va OOS bo'sh joylarini chizadi.

    Belgilar:
      rangli quti + nom   — tanilgan SKU
      kulrang "?cN"       — noma'lum, N-klaster (operator bir nom berishi mumkin)
      qizil to'ldirilgan  — OOS (bo'sh joy)
      to'q sariq punktir  — shubhali bo'shliq (detektor o'tkazib yuborgan bo'lishi mumkin)
      ko'k punktir        — barcha qatorlarda takrorlangan bo'shliq: bo'linma yoki butun ustun OOS
      ingichka kulrang    — orqa fon (KPI'ga kirmagan)
    """
    if dim and result.roi is not None:
        image = dim_outside(image, [result.roi])
    img = Image.fromarray(image.copy()).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    scale = max(1.0, img.width / 1400)
    font = _font(int(13 * scale))
    lw = max(2, int(2 * scale))

    for d in result.background:
        draw.rectangle(list(d.box.as_xyxy()), outline=(120, 120, 120, 160), width=1)

    for g in result.gaps:
        draw.rectangle([g.x1, g.y1, g.x2, g.y2], fill=(220, 40, 40, 70),
                       outline=(220, 40, 40, 255), width=lw)
        draw.text((g.x1 + 3, g.y1 + 3), f"OOS x{g.est_missing_facings}",
                  fill=(255, 255, 255), font=font)
    for g in result.dividers:
        _dashed_rect(draw, (g.x1, g.y1, g.x2, g.y2), (70, 130, 230, 255), width=1, dash=5)
        draw.text((g.x1 + 3, g.y1 + 3), "bo'linma?", fill=(70, 130, 230), font=font)
    for g in result.suspects:
        _dashed_rect(draw, (g.x1, g.y1, g.x2, g.y2), (255, 150, 0, 255), width=lw)
        draw.text((g.x1 + 3, g.y1 + 3), "?OOS", fill=(255, 150, 0), font=font)

    for d in result.detections:
        x1, y1, x2, y2 = d.box.as_xyxy()
        if d.match and d.match.is_known:
            color = _color_for(d.match.sku_id)
            label = d.match.name or d.match.sku_id
            if show_scores:
                label += f" {d.match.score:.2f}"
            if d.context_fixed:
                label += "*"
        elif d.match and d.match.level == "brand":
            color = _color_for(f"brand:{d.match.brand}")
            label = f"{d.match.brand} (SKU?)"
        elif d.match and d.match.level == "variant" and d.match.candidates:
            color = _color_for(f"brand:{d.match.brand or d.match.candidates[0].get('sku_id')}")
            label = f"{d.match.candidates[0].get('name')} (rang?)"
        elif d.cluster >= 0:
            color = _color_for(f"cluster{d.cluster}")
            color = tuple(int(0.5 * c + 0.5 * 150) for c in color)
            label = f"?c{d.cluster}"
        else:
            color = (150, 150, 150)
            label = "?"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 4)
        draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
        draw.text((x1 + 3, ly + 1), label, fill=(20, 20, 20), font=font)

    if result.roi is not None:
        _draw_roi(draw, result.roi, _font(int(16 * scale)), lw)
    return np.array(img)


def cluster_montage(image: np.ndarray, result: AnalysisResult, thumb=(80, 150),
                    max_per_cluster: int = 12) -> np.ndarray | None:
    """Noma'lum klasterlar montaji — operator har qatorga bitta nom beradi."""
    groups: dict[int, list] = {}
    for i, d in enumerate(result.detections):
        if d.cluster >= 0 and not (d.match and d.match.is_known):
            groups.setdefault(d.cluster, []).append(d)
    if not groups:
        return None
    tw, th = thumb
    pad, label_w = 4, 70
    cols = min(max_per_cluster, max(len(v) for v in groups.values()))
    W = label_w + cols * (tw + pad)
    H = len(groups) * (th + pad)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    src = Image.fromarray(image)
    for r, (cid, ds) in enumerate(sorted(groups.items())):
        y = r * (th + pad)
        draw.text((6, y + th // 2 - 8), f"c{cid} ({len(ds)})", fill=(0, 0, 0), font=font)
        for k, d in enumerate(ds[:max_per_cluster]):
            crop = src.crop(d.box.as_xyxy())
            crop.thumbnail((tw, th))
            canvas.paste(crop, (label_w + k * (tw + pad), y))
    return np.array(canvas)
