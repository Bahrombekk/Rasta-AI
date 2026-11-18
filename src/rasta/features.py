"""Qo'shimcha (klassik) deskriptorlar: RANG va SHAKL.

Nega kerak? DINOv2 embedding'i umumiy ko'rinishni (tekstura, naqsh, shakl)
yaxshi ushlaydi, lekin rangga nisbatan ancha "befarq": qora Chin8 bankasi va
to'q sariq Fanta bankasi bir xil shakl va kompozitsiyaga ega bo'lgani uchun
DINOv2 ularni yaqin deb hisoblashi mumkin. Ichimlik/FMCG brendlarida esa rang
— eng kuchli farqlovchi belgi (Coca-Cola qizil, Pepsi ko'k, Fanta to'q sariq).

Shuning uchun tanish GIBRID: yakuniy ball = DINOv2 + rang + shakl.

Rang deskriptori (yorug'likka chidamli):
  * HSV fazosi; to'yingan piksellar -> rang tusi (hue) bo'yicha yumshoq binlar,
    to'yinmagan piksellar -> qora / kulrang / oq binlar (axromatik).
  * Fazoviy: butun crop + 3 gorizontal tasma (tepa/o'rta/past) — qopqoq va
    yorliq ranglarining joylashuvi ham farqlaydi.
  * Hellinger normallashtirish (sqrt + L2): ikki vektor skalyar ko'paytmasi =
    Bhattacharyya koeffitsienti, [0, 1] oralig'ida.

Shakl deskriptori: bbox tomonlar nisbati (h/w) — 0.33L banka va 1.5L shisha
yoki quti va bankani ajratadi.
"""
from __future__ import annotations

import numpy as np

HUE_BINS = 16
ACHRO_BINS = 3                      # qora, kulrang, oq
BINS = HUE_BINS + ACHRO_BINS
STRIPES = 3                         # tepa / o'rta / past
COLOR_DIM = BINS * (1 + STRIPES)

# Axromatik chegaralar (OpenCV HSV: S,V 0..255)
SAT_MIN = 55
DARK_V = 50
WHITE_V = 175


def _hist(hsv: np.ndarray) -> np.ndarray:
    """(N,3) HSV piksellar -> BINS o'lchamli (normallashtirilmagan) gistogramma."""
    h = hsv[:, 0].astype(np.float32)   # 0..179
    s = hsv[:, 1]
    v = hsv[:, 2]
    out = np.zeros(BINS, dtype=np.float32)

    dark = v < DARK_V
    chroma = (~dark) & (s >= SAT_MIN)
    achro = (~dark) & (s < SAT_MIN)

    # Hue — yumshoq (chiziqli interpolyatsiya) binlash: yorug'lik siljishiga chidamli
    if chroma.any():
        pos = h[chroma] / 180.0 * HUE_BINS - 0.5
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        # to'yinganlik bilan og'irlash: yorqin ranglar kuchliroq hissa qo'shadi
        w = (s[chroma].astype(np.float32) / 255.0)
        np.add.at(out, lo % HUE_BINS, (1.0 - frac) * w)
        np.add.at(out, (lo + 1) % HUE_BINS, frac * w)
        # og'irlik yo'qotilgan qismi (1-w) — kulrangga yaqin, uni axromatikka qo'shamiz
        out[HUE_BINS + 1] += float((1.0 - w).sum())

    out[HUE_BINS + 0] += float(dark.sum())
    if achro.any():
        va = v[achro]
        out[HUE_BINS + 1] += float((va < WHITE_V).sum())
        out[HUE_BINS + 2] += float((va >= WHITE_V).sum())
    return out


def color_descriptor(crop: np.ndarray) -> np.ndarray:
    """RGB crop -> COLOR_DIM o'lchamli Hellinger-normallashgan rang vektori."""
    import cv2

    if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        v = np.zeros(COLOR_DIM, dtype=np.float32)
        v[HUE_BINS + 1 :: BINS] = 1.0
        return v / np.linalg.norm(v)

    h, w = crop.shape[:2]
    # Markaziy qism: chetdagi fon / qo'shni mahsulot ta'sirini kamaytiradi
    x1, x2 = int(w * 0.12), max(int(w * 0.88), int(w * 0.12) + 1)
    y1, y2 = int(h * 0.04), max(int(h * 0.96), int(h * 0.04) + 1)
    c = crop[y1:y2, x1:x2]
    # Tezlik uchun kichraytirish
    ch, cw = c.shape[:2]
    scale = min(1.0, 96.0 / max(ch, cw))
    if scale < 1.0:
        c = cv2.resize(c, (max(2, int(cw * scale)), max(2, int(ch * scale))),
                       interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_RGB2HSV)

    parts = []
    flat = hsv.reshape(-1, 3)
    g = _hist(flat)
    parts.append(g / max(g.sum(), 1e-6))
    H = hsv.shape[0]
    for k in range(STRIPES):
        a, b = H * k // STRIPES, max(H * (k + 1) // STRIPES, H * k // STRIPES + 1)
        hk = _hist(hsv[a:b].reshape(-1, 3))
        parts.append(hk / max(hk.sum(), 1e-6))

    # Global gistogramma tasmalarga teng og'irlikda (1 : 1/3 : 1/3 : 1/3)
    weights = [1.0] + [1.0 / STRIPES] * STRIPES
    vec = np.concatenate([np.sqrt(p * wt) for p, wt in zip(parts, weights)])
    vec = vec.astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-6)


_POOL = None


def color_descriptors(crops: list[np.ndarray]) -> np.ndarray:
    global _POOL
    if not crops:
        return np.zeros((0, COLOR_DIM), dtype=np.float32)
    if len(crops) < 16:
        return np.stack([color_descriptor(c) for c in crops])
    if _POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
    return np.stack(list(_POOL.map(color_descriptor, crops)))


def color_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Ikki rang vektori o'xshashligi (Bhattacharyya), 0..1."""
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def aspect_ratio(height: float, width: float) -> float:
    return float(height) / max(float(width), 1e-6)


def shape_similarity(ar_a: float, ar_b: float, sigma: float = 0.25) -> float:
    """Tomonlar nisbati o'xshashligi: exp(-(ln(a/b))^2 / 2σ^2), 0..1."""
    if ar_a <= 0 or ar_b <= 0:
        return 1.0  # ma'lumot yo'q -> jarima yo'q
    d = np.log(ar_a / ar_b)
    return float(np.exp(-(d * d) / (2.0 * sigma * sigma)))


def to_rgb_array(img) -> np.ndarray:
    """PIL yoki numpy -> RGB uint8 numpy."""
    from PIL import Image

    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)
