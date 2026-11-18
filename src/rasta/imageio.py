"""Rasm yuklash/saqlash yordamchilari. Ichkarida hamma joyda RGB numpy ishlatiladi."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Fayldan RGB (H,W,3) uint8 numpy massiv sifatida yuklaydi."""
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def save_image(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
