"""Sintetik javon generatori — internetsiz demo va test uchun.

Hisobotning 16.1-bo'limidagi "copy-paste" g'oyasining soddalashtirilgan
ko'rinishi: mahsulot "packshot"lari yaratiladi va javon foniga joylanadi.
Bir nechta bo'sh joy (OOS) qoldiriladi. Butun pipeline (detektor -> embedding
-> qidiruv -> OOS -> KPI) shu bilan haqiqiy modellarda sinaladi.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Box


@dataclass
class SkuDef:
    sku_id: str
    name: str
    brand: str
    body: tuple[int, int, int]
    cap: tuple[int, int, int]
    shape: str  # "bottle" | "can" | "box"


# 8 xil SKU (3 brend) — o'xshash lekin farqlanadigan
PRODUCTS: list[SkuDef] = [
    SkuDef("cola_05", "Cola 0.5L", "ColaCo", (200, 30, 40), (40, 40, 40), "bottle"),
    SkuDef("cola_10", "Cola 1.0L", "ColaCo", (170, 25, 35), (200, 200, 200), "bottle"),
    SkuDef("fanta_05", "Orange 0.5L", "ColaCo", (240, 140, 20), (40, 40, 40), "bottle"),
    SkuDef("water_05", "Water 0.5L", "AquaBrand", (60, 160, 220), (30, 90, 160), "bottle"),
    SkuDef("water_15", "Water 1.5L", "AquaBrand", (90, 190, 230), (255, 255, 255), "bottle"),
    SkuDef("juice_can", "Juice Can", "FreshCo", (230, 200, 40), (180, 150, 20), "can"),
    SkuDef("energy_can", "Energy Can", "FreshCo", (40, 190, 120), (20, 120, 80), "can"),
    SkuDef("choco_box", "Choco Box", "FreshCo", (110, 70, 40), (70, 45, 25), "box"),
]

BRAND_OF = {p.sku_id: p.brand for p in PRODUCTS}

# Qiyin holatlar (stress-test uchun):
#  * RANG-egizaklar: shakl bir xil, faqat rang farq qiladi (DINOv2 uchun qiyin)
#  * SHAKL-egizaklar: rang bir xil, shakl/hajm farq qiladi
HARD_PRODUCTS: list[SkuDef] = [
    SkuDef("lime_05", "Lime 0.5L", "ColaCo", (70, 170, 50), (40, 40, 40), "bottle"),
    SkuDef("grape_05", "Grape 0.5L", "ColaCo", (110, 40, 130), (40, 40, 40), "bottle"),
    SkuDef("cola_can", "Cola Can", "ColaCo", (200, 30, 40), (40, 40, 40), "can"),
    SkuDef("water_can", "Water Can", "AquaBrand", (60, 160, 220), (30, 90, 160), "can"),
    SkuDef("berry_can", "Berry Can", "FreshCo", (200, 40, 110), (120, 20, 60), "can"),
    SkuDef("mint_box", "Mint Box", "FreshCo", (40, 190, 120), (20, 120, 80), "box"),
]

# Katalogda YO'Q mahsulotlar (open-set: tizim ularni "noma'lum" deyishi kerak)
UNKNOWN_PRODUCTS: list[SkuDef] = [
    SkuDef("unk_black_can", "Black Can", "Other", (25, 25, 30), (200, 170, 40), "can"),
    SkuDef("unk_pink_bottle", "Pink Bottle", "Other", (240, 120, 180), (255, 255, 255), "bottle"),
    SkuDef("unk_gray_box", "Gray Box", "Other", (130, 130, 135), (60, 60, 60), "box"),
    SkuDef("unk_teal_can", "Teal Can", "Other", (20, 140, 140), (230, 230, 230), "can"),
]

ALL_PRODUCTS = PRODUCTS + HARD_PRODUCTS
BRAND_OF_ALL = {p.sku_id: p.brand for p in ALL_PRODUCTS + UNKNOWN_PRODUCTS}


def _stable_seed(key: str) -> int:
    """PYTHONHASHSEED'ga bog'liq bo'lmagan barqaror seed (hash() o'rniga)."""
    return zlib.crc32(key.encode("utf-8")) % 999


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_product(sku: SkuDef, w: int, h: int, seed: int | None = None) -> np.ndarray:
    """Bitta mahsulotni shaffof fonda chizadi (RGBA)."""
    rng = random.Random(seed)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def jit(c, amt=12):
        return tuple(max(0, min(255, ci + rng.randint(-amt, amt))) for ci in c)

    body = jit(sku.body)
    cap = jit(sku.cap)
    m = int(w * 0.12)  # chetki bo'shliq

    if sku.shape == "bottle":
        neck_w = int(w * 0.32)
        cx = w // 2
        # qopqoq
        d.rectangle([cx - neck_w // 2, m, cx + neck_w // 2, int(h * 0.16)], fill=cap)
        # tana
        d.rounded_rectangle([m, int(h * 0.16), w - m, h - m],
                            radius=int(w * 0.18), fill=body)
        # yorliq (oq band)
        d.rectangle([m + 2, int(h * 0.45), w - m - 2, int(h * 0.72)],
                    fill=(245, 245, 245))
    elif sku.shape == "can":
        d.rounded_rectangle([m, m, w - m, h - m], radius=int(w * 0.12), fill=body)
        d.rectangle([m, m, w - m, int(h * 0.12)], fill=cap)  # tepa
        d.rectangle([m + 2, int(h * 0.40), w - m - 2, int(h * 0.66)],
                    fill=(245, 245, 245))
    else:  # box
        d.rectangle([m, m, w - m, h - m], fill=body)
        d.rectangle([m, m, w - m, int(h * 0.30)], fill=cap)
        d.rectangle([m + 2, int(h * 0.42), w - m - 2, int(h * 0.70)],
                    fill=(245, 245, 245))

    # brend harfi (yorliq ustida)
    f = _font(max(10, int(w * 0.28)))
    letter = sku.brand[0]
    tb = d.textbbox((0, 0), letter, font=f)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    d.text(((w - tw) / 2, int(h * 0.50)), letter, fill=body, font=f)

    return np.array(img)


def _paste_rgb(bg: Image.Image, prod_rgba: np.ndarray, x: int, y: int):
    p = Image.fromarray(prod_rgba, "RGBA")
    bg.paste(p, (x, y), p)


def generate_catalog(
    variants: int = 5, products: list[SkuDef] | None = None
) -> dict[str, list[np.ndarray]]:
    """Har SKU uchun bir nechta packshot varianti (augmentatsiya bilan) qaytaradi.

    Bu katalog onboarding'ni simulyatsiya qiladi: yangi mahsulot = bir nechta rasm.
    """
    catalog: dict[str, list[np.ndarray]] = {}
    for p in (products or PRODUCTS):
        imgs = []
        for v in range(variants):
            if p.shape == "bottle":
                w, h = 140, 300
            else:
                w, h = 140, 200
            rgba = render_product(p, w, h, seed=1000 + _stable_seed(p.sku_id) + v)
            # oq fonga qo'yamiz (packshot)
            bg = Image.new("RGB", (w, h), (255, 255, 255))
            _paste_rgb(bg, rgba, 0, 0)
            arr = np.array(bg)
            imgs.append(arr)
        catalog[p.sku_id] = imgs
    return catalog


def generate_shelf(
    rows: int = 3,
    cols: int = 8,
    cell_w: int = 150,
    cell_h: int = 320,
    gap: int = 14,
    empty_slots: int = 4,
    seed: int = 42,
    products: list[SkuDef] | None = None,
    unknown_products: list[SkuDef] | None = None,
    unknown_frac: float = 0.0,
    block_size: int = 1,
    return_empties: bool = False,
    empty_rows: tuple[int, ...] = (),
):
    """Javon rasmi + haqiqiy (ground-truth) qutilar + SKU yorliqlari qaytaradi.

    Ba'zi kataklar bo'sh qoldiriladi (OOS testi uchun). unknown_frac ulushda
    katalogda yo'q mahsulotlar qo'yiladi (yorlig'i "unknown:<sku>").
    block_size>1 — bir SKU yonma-yon bloklarda (real merchandayzing kabi).
    return_empties=True -> (img, boxes, labels, empties) qaytaradi.
    """
    rng = random.Random(seed)
    products = products or PRODUCTS
    unknown_products = unknown_products or UNKNOWN_PRODUCTS
    margin = 30
    shelf_thick = 16
    W = margin * 2 + cols * cell_w + (cols - 1) * gap
    H = margin * 2 + rows * cell_h + (rows - 1) * (gap + shelf_thick)

    bg = Image.new("RGB", (W, H), (225, 222, 214))
    d = ImageDraw.Draw(bg)

    # bo'sh kataklarni tanlash
    all_slots = [(r, c) for r in range(rows) for c in range(cols)]
    empties = set(rng.sample(all_slots, min(empty_slots, len(all_slots))))
    empties |= {(r, c) for r in empty_rows for c in range(cols)}   # butunlay bo'sh qator

    boxes: list[Box] = []
    labels: list[str] = []
    prev = None

    for r in range(rows):
        y0 = margin + r * (cell_h + gap + shelf_thick)
        # javon taxtasi
        shelf_y = y0 + cell_h
        d.rectangle([margin - 10, shelf_y, W - margin + 10, shelf_y + shelf_thick],
                    fill=(150, 120, 85))
        prev = None
        for c in range(cols):
            if (r, c) in empties:
                continue
            x0 = margin + c * (cell_w + gap)
            if c % max(1, block_size) == 0 or prev is None:
                if unknown_frac > 0 and rng.random() < unknown_frac:
                    prev = ("unknown", rng.choice(unknown_products))
                else:
                    prev = ("known", rng.choice(products))
            kind, sku = prev
            pw = int(cell_w * rng.uniform(0.80, 0.95))
            ph = int(cell_h * (0.95 if sku.shape == "bottle" else rng.uniform(0.60, 0.72)))
            px = x0 + (cell_w - pw) // 2
            py = shelf_y - ph  # javon taxtasi ustida turadi
            rgba = render_product(sku, pw, ph, seed=rng.randint(0, 10**6))
            _paste_rgb(bg, rgba, px, py)
            boxes.append(Box(px, py, px + pw, py + ph, 1.0))
            labels.append(sku.sku_id if kind == "known" else f"unknown:{sku.sku_id}")

    if return_empties:
        return np.array(bg), boxes, labels, sorted(empties)
    return np.array(bg), boxes, labels


def distort(image: np.ndarray, seed: int = 0, strength: float = 1.0) -> np.ndarray:
    """Real kamera sharoitini taqlid qiladi: yorug'lik, rang harorati, blur,
    shovqin, JPEG siqish, vinyetka. Geometriya (qutilar) o'zgarmaydi."""
    import io

    from PIL import ImageEnhance, ImageFilter

    rng = random.Random(seed)
    im = Image.fromarray(image)
    s = strength
    im = ImageEnhance.Brightness(im).enhance(1.0 + rng.uniform(-0.35, 0.25) * s)
    im = ImageEnhance.Contrast(im).enhance(1.0 + rng.uniform(-0.3, 0.2) * s)
    im = ImageEnhance.Color(im).enhance(1.0 + rng.uniform(-0.3, 0.2) * s)
    arr = np.asarray(im).astype(np.float32)
    # rang harorati (oq balans xatosi)
    t = rng.uniform(-1, 1) * 18 * s
    arr[..., 0] += t
    arr[..., 2] -= t
    # vinyetka
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    arr *= (1.0 - 0.25 * s * np.clip(r - 0.4, 0, 1))[..., None]
    # shovqin
    nrng = np.random.default_rng(seed)
    arr += nrng.normal(0, 6 * s, arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.7:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6) * s))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(rng.uniform(55, 90)))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))
