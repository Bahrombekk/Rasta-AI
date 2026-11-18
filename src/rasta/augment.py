"""Few-shot onboarding uchun augmentatsiya: 1 ta rasm -> N ta realistik variant.

Ko'pincha yangi SKU uchun faqat bitta toza packshot bo'ladi, javondagi crop esa
boshqacha: yorug'lik, oq balans, blur, qisman kesilgan chet, JPEG siqilishi.
Katalogga shu sharoitlarni taqlid qiluvchi variantlar qo'shilsa, packshot va
real crop orasidagi "domen farqi" kamayadi.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .features import to_rgb_array


def make_variants(img, n: int = 6, seed: int = 0) -> list[np.ndarray]:
    """Asl rasm + (n-1) ta tasodifiy variant (deterministik seed bilan)."""
    base = Image.fromarray(to_rgb_array(img))
    out = [np.array(base)]
    rng = random.Random(seed)
    for _ in range(max(0, n - 1)):
        im = base
        w, h = im.size
        # kichik kesish/siljish (detektor qutisi noaniqligi)
        dx1, dx2 = rng.uniform(0, 0.08) * w, rng.uniform(0, 0.08) * w
        dy1, dy2 = rng.uniform(0, 0.06) * h, rng.uniform(0, 0.06) * h
        im = im.crop((int(dx1), int(dy1), max(int(dx1) + 2, int(w - dx2)),
                      max(int(dy1) + 2, int(h - dy2))))
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.2))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.15))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.8, 1.15))
        arr = np.asarray(im).astype(np.float32)
        t = rng.uniform(-12, 12)                     # oq balans xatosi
        arr[..., 0] += t
        arr[..., 2] -= t
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if rng.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        if rng.random() < 0.5:                       # past o'lcham (uzoqdan suratga olingan)
            s = rng.uniform(0.35, 0.7)
            im = im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))),
                           Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=rng.randint(50, 90))
        buf.seek(0)
        out.append(np.array(Image.open(buf).convert("RGB")))
    return out


def packshot_crop(img, tol: int = 18, min_border_frac: float = 0.9, pad_frac: float = 0.03,
                  max_fill: float = 0.6):
    """Bir xil (odatda oq) fonli katalog rasmini mahsulot chegarasigacha kesadi.

    Internetdan olingan packshot'larda mahsulot ko'pincha rasmning kichik qismini
    egallaydi (masalan 447x447 kvadratda ~15%). Model butun rasmni ko'rgani uchun
    javondagi zich kesilgan crop bilan o'xshashlik pasayadi. Qaytaradi:
    (kesilgan rasm, packshot_mi). Fon bir xil bo'lmasa — rasm o'zgarmaydi.
    """
    a = to_rgb_array(img)
    h, w = a.shape[:2]
    if h < 16 or w < 16:
        return a, False
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(np.int16)
    bg = np.median(border, axis=0)
    close = (np.abs(border - bg).max(axis=1) <= tol).mean()
    if close < min_border_frac:
        return a, False                      # fon bir xil emas — real foto/crop
    diff = np.abs(a.astype(np.int16) - bg).max(axis=2) > tol
    ys, xs = np.where(diff)
    if len(xs) < 0.01 * h * w:
        return a, False
    x1, x2 = np.percentile(xs, [0.2, 99.8]).astype(int)
    y1, y2 = np.percentile(ys, [0.2, 99.8]).astype(int)
    px, py = int((x2 - x1) * pad_frac), int((y2 - y1) * pad_frac)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px + 1), min(h, y2 + py + 1)
    if (x2 - x1) * (y2 - y1) > max_fill * w * h:
        # mahsulot rasmning katta qismini egallagan — kesish foyda bermaydi
        # (Grocery benchmark: allaqachon zich kadrlangan rasmlarni kesish zarar qiladi)
        return a, True
    return np.ascontiguousarray(a[y1:y2, x1:x2]), True
