"""Namuna (katalog) fotosidagi asosiy mahsulotni topish.

Zich javonga o'qitilgan detektor yaqindan olingan BITTA mahsulotni ishonchli topa olmaydi
(6 ta real namunadan 1-2 tasi; yon tomondan olinganlar umuman topilmadi). Matn so'rovli
detektor (Grounding DINO) esa aynan shunday sahnada kuchli: 6/6. U faqat katalogga rasm
qo'shilganda ishlaydi (javon tahlilida emas), shuning uchun birinchi murojaatda yuklanadi.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from .augment import pick_product_box
from .features import to_rgb_array

log = logging.getLogger(__name__)
DEFAULT_PROMPTS = ("product package", "bottle", "can", "carton", "box")


class ProductLocator:
    def __init__(self, model_id: str, device: str, prompts=DEFAULT_PROMPTS, conf: float = 0.25,
                 max_side: int = 800):
        from .detector import ZeroShotDetector

        self.detector = ZeroShotDetector(model_id, device, prompts=tuple(prompts), conf=conf,
                                         text_conf=0.2, max_area=0.95)
        self.max_side = max_side

    def locate(self, image) -> Optional[tuple[int, int, int, int]]:
        """Asl rasm pikselidagi (x1,y1,x2,y2) yoki None."""
        a = to_rgb_array(image)
        h, w = a.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = a if scale == 1.0 else np.array(
            Image.fromarray(a).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR))
        sh, sw = small.shape[:2]
        box = pick_product_box(self.detector.detect(small), sw, sh)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return (int(x1 / scale), int(y1 / scale), min(w, int(round(x2 / scale))), min(h, int(round(y2 / scale))))


def build_locator(cfg, device: str) -> Optional[ProductLocator]:
    """onboarding.locator sozlamasi bo'yicha; model yo'q/yuklanmasa None (zaxira yo'l ishlaydi)."""
    spec = cfg.get("onboarding.locator") or {}
    model = spec.get("model")
    if not model:
        return None
    try:
        return ProductLocator(model, device, prompts=spec.get("prompts") or DEFAULT_PROMPTS,
                              conf=spec.get("conf", 0.25), max_side=spec.get("max_side", 800))
    except Exception as exc:                # model keshda yo'q (oflayn image) va h.k.
        log.warning("Mahsulot lokatori yuklanmadi (%s): %s — javon detektori ishlatiladi", model, exc)
        return None
