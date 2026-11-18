"""HTTP transport for model inference. Catalog and shelf logic run in the caller."""
from __future__ import annotations

import io
import os

import numpy as np
import requests
from PIL import Image

from .catalog import embedder_signature
from .features import to_rgb_array
from .types import Box

IMAGE_BATCH = 32


class InferenceUnavailable(RuntimeError):
    pass


class RemoteInference:
    def __init__(self, cfg, url: str, token: str | None = None):
        token = token or os.environ.get("RASTA_AI_TOKEN", "")
        if len(token) < 32:
            raise ValueError("RASTA_AI_TOKEN must contain at least 32 characters")
        self.url = url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["X-Rasta-AI-Token"] = token
        self.signature = embedder_signature(cfg)
        info = self._info()
        self.dim = info["dim"]
        self.device = info["device"]

    def _request(self, method, path, timeout, **kwargs):
        try:
            response = self.session.request(method, self.url + path, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise InferenceUnavailable("AI xizmatiga ulanib bo‘lmadi.") from exc

    def _info(self):
        info = self._request("GET", "/health", (3, 3))
        if info.get("signature") != self.signature:
            raise InferenceUnavailable("Backend va AI embedding sozlamalari bir xil bo‘lishi kerak.")
        return info

    def ready(self):
        try:
            self._info()
            return True
        except InferenceUnavailable:
            return False

    @staticmethod
    def _file(image):
        with io.BytesIO() as buffer:
            Image.fromarray(to_rgb_array(image)).save(buffer, format="PNG", compress_level=1)
            return "image.png", buffer.getvalue(), "image/png"

    def detect(self, image):
        data = self._request("POST", "/detect", (3, 180), files={"image": self._file(image)})
        return [Box(*box) for box in data["boxes"]]

    def embed(self, images):
        batches = []
        for start in range(0, len(images), IMAGE_BATCH):
            files = [("images", self._file(image)) for image in images[start:start + IMAGE_BATCH]]
            data = self._request("POST", "/embed", (3, 180), files=files)
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            if vectors.shape != (len(files), self.dim) or not np.isfinite(vectors).all():
                raise InferenceUnavailable("AI xizmati noto‘g‘ri vektor qaytardi.")
            batches.append(vectors)
        return np.concatenate(batches) if batches else np.empty((0, self.dim), dtype=np.float32)
