"""Private service that reads the text printed on product labels.

Kept apart from the AI service: PaddleOCR brings its own framework and OpenCV build, which
must not share an environment with PyTorch and Ultralytics. Runs on the GPU when
`RASTA_OCR_DEVICE` names one (`gpu:0`) and the installed Paddle build and the container can
see it; otherwise it falls back to the CPU, with the same models and the same results.
"""
from __future__ import annotations

import hmac
import io
import os
import threading

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image

MAX_IMAGES = 16
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 9_000_000
# Measured on shelf crops (BENCHMARK.md 11f): the mobile detector reads as many labels as the
# server one at a tenth of the CPU time; the Cyrillic-specific recogniser was not better.
DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "PP-OCRv6_medium_rec"


class PaddleReader:
    def __init__(self, threads: int | None = None, device: str | None = None):
        wanted = (device or os.environ.get("RASTA_OCR_DEVICE", "cpu")).strip().lower() or "cpu"
        self.device = wanted
        try:
            self.ocr = self._build(wanted, threads)
            self._warm_up()                        # a missing GPU often shows only at the first inference
        except Exception as error:                 # CPU build of Paddle, no GPU in the container, driver mismatch
            if wanted == "cpu":
                raise
            print(f"[rasta-ocr] {wanted} unavailable ({type(error).__name__}: {str(error)[:160]}) — using the CPU",
                  flush=True)
            self.device = "cpu"
            self.ocr = self._build("cpu", threads)
            self._warm_up()
        print(f"[rasta-ocr] reading on {self.device}", flush=True)

    def _warm_up(self) -> None:
        """One read of a blank image: loads the weights onto the device, so the first real request is
        not slow and a broken device fails here instead of in the middle of an analysis."""
        self.read(np.full((96, 160, 3), 255, np.uint8))

    @staticmethod
    def _build(device: str, threads: int | None):
        from paddleocr import PaddleOCR
        if device != "cpu":
            import paddle
            if not paddle.device.is_compiled_with_cuda():
                raise RuntimeError("this PaddlePaddle build has no CUDA support")
        return PaddleOCR(
            text_detection_model_name=DET_MODEL, text_recognition_model_name=REC_MODEL,
            use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
            device=device,
            # oneDNN fails on these models in paddlepaddle 3.3 ("ConvertPirAttribute2RuntimeAttribute")
            enable_mkldnn=False, cpu_threads=threads or int(os.environ.get("RASTA_OCR_THREADS", "8")))

    def read(self, pixels: np.ndarray) -> list[list]:
        """RGB image -> [[text, confidence, x1, x2, size], ...]; x1, x2 are the text's left and
        right edges as a fraction of the image width, size is the letter height as a fraction of
        the image's shorter side (a brand name is large, the ingredients are not)."""
        height, width = pixels.shape[:2]
        rows = []
        for page in self.ocr.predict(np.ascontiguousarray(pixels[:, :, ::-1])):
            for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
                if text.strip():
                    points = np.asarray(poly, dtype=float)
                    xs = points[:, 0]
                    sides = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
                    rows.append([text.strip(), round(float(score), 3),
                                 round(float(xs.min()) / width, 3), round(float(xs.max()) / width, 3),
                                 round(float(sides.min()) / min(height, width), 3)])
        return rows


def read_image(file: UploadFile) -> np.ndarray:
    raw = file.file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the OCR limit")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Image exceeds the OCR pixel limit")
            return np.array(image.convert("RGB"))
    except (OSError, Image.DecompressionBombError, ValueError) as exc:
        raise HTTPException(400, "Invalid OCR image") from exc


def create_ocr_app(token: str, *, reader=None):
    if len(token) < 32:
        raise ValueError("RASTA_AI_TOKEN must contain at least 32 characters")
    reader = PaddleReader() if reader is None else reader
    lock = threading.Lock()

    def authenticate(x_rasta_ai_token: str = Header("")):
        if not hmac.compare_digest(x_rasta_ai_token.encode(), token.encode()):
            raise HTTPException(401, "OCR service authentication required")

    app = FastAPI(title="Rasta OCR service", docs_url=None, redoc_url=None)

    @app.get("/ready")
    def readiness():
        return {"ready": True, "device": getattr(reader, "device", "cpu")}

    @app.post("/read", dependencies=[Depends(authenticate)])
    def read(images: list[UploadFile] = File(...)):
        if not 1 <= len(images) <= MAX_IMAGES:
            raise HTTPException(400, f"Send 1–{MAX_IMAGES} images")
        pixels = [read_image(image) for image in images]
        with lock:
            results = [reader.read(p) for p in pixels]
        return {"results": results}

    return app
