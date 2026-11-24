"""Private service that owns detector and embedding models."""
from __future__ import annotations

import hmac
import io
import threading

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

from rasta.catalog import embedder_signature
from rasta.config import resolve_device
from rasta.augment import pick_product_box
from rasta.detector import build_detector, suppress_nested
from rasta.embedder import build_embedder
from rasta.inference import IMAGE_BATCH

MAX_IMAGE_BYTES = 32 * 1024 * 1024


def read_image(file: UploadFile):
    raw = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the inference limit")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 9_000_000:
                raise HTTPException(413, "Image exceeds the inference pixel limit")
            return np.array(image.convert("RGB"))
    except (OSError, Image.DecompressionBombError, ValueError) as exc:
        raise HTTPException(400, "Invalid inference image") from exc


def create_ai_app(cfg, token, *, detector=None, embedder=None, locator=None):
    if len(token) < 32:
        raise ValueError("RASTA_AI_TOKEN must contain at least 32 characters")
    device = resolve_device(cfg.get("device", "auto")) if detector is None or embedder is None else "test"
    detector = build_detector(cfg, device) if detector is None else detector
    embedder = build_embedder(cfg, device) if embedder is None else embedder
    info = {"dim": embedder.dim, "device": device, "signature": embedder_signature(cfg)}
    lock = threading.Lock()
    state = {"locator": locator}        # None — hali yuklanmagan; False — mavjud emas

    def authenticate(x_rasta_ai_token: str = Header("")):
        if not hmac.compare_digest(x_rasta_ai_token.encode(), token.encode()):
            raise HTTPException(401, "AI service authentication required")

    app = FastAPI(title="Rasta AI service", docs_url=None, redoc_url=None)

    @app.get("/ready")
    def readiness():
        return {"ready": True}

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health():
        return info

    @app.post("/detect", dependencies=[Depends(authenticate)])
    def detect(image: UploadFile = File(...)):
        pixels = read_image(image)
        with lock:
            boxes = detector.detect(pixels)
        return {"boxes": [[float(b.x1), float(b.y1), float(b.x2), float(b.y2), float(b.score)]
                          for b in boxes]}

    @app.post("/locate", dependencies=[Depends(authenticate)])
    def locate(image: UploadFile = File(...)):
        """Katalog namunasi fotosidagi asosiy mahsulot qutisi (piksel) yoki null."""
        pixels = read_image(image)
        with lock:
            if state["locator"] is None:
                from rasta.locator import build_locator
                state["locator"] = build_locator(cfg, device) or False
            if state["locator"]:
                box = state["locator"].locate(pixels)
            else:                           # lokator modeli image'da yo'q — javon detektori
                h, w = pixels.shape[:2]
                found = suppress_nested(detector.detect(pixels), cfg.get("detector.contain_thr", 0.8))
                box = pick_product_box(found, w, h)
        return {"box": [int(v) for v in box] if box else None}

    @app.post("/embed", dependencies=[Depends(authenticate)])
    def embed(images: list[UploadFile] = File(...)):
        if not 1 <= len(images) <= IMAGE_BATCH:
            raise HTTPException(400, f"Send 1–{IMAGE_BATCH} images")
        pixels = [read_image(image) for image in images]
        with lock:
            vectors = embedder.embed(pixels)
        return {"vectors": vectors.tolist()}

    return app
