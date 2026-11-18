"""Embedding moduli — mahsulot crop'ini vektorga aylantiradi.

Embedding modeli "bu Coca-Cola" deb o'rganmaydi; u "ikki rasm bir xilmi"
degan o'xshashlikni o'lchaydi. Shu sabab yangi SKU modelni qayta o'qitishni
talab qilmaydi — faqat katalogga vektor qo'shiladi.

Qo'llab-quvvatlanadigan backbone'lar (embedder.model):
  facebook/dinov2-{small,base,large}   — sof vizual (tekstura, naqsh, shakl)
  openai/clip-vit-large-patch14        — vizual + matn/logotip "o'qish"
  google/siglip2-*                     — CLIP oilasi, retrieval uchun kuchli
  "A + B"                              — ansambl: L2-normallashgan vektorlar
                                         og'irlik bilan birlashtiriladi
                                         (embedder.weights: [wA, wB])

Muhim tafsilot — preprocessing:
  HuggingFace protsessori standart holatda kichik tomonni 256 ga keltirib,
  markazdan 224x224 kesadi. Baland mahsulot (banka/shisha, h/w ~ 2.5) uchun bu
  faqat o'rta qismni qoldiradi — qopqoq va pastki yorliq yo'qoladi. Shuning
  uchun bu yerda o'z preprocessing'imiz bor:
    resize_mode = "pad"     -> tomonlar nisbati saqlanadi, kvadratga to'ldiriladi
                  "stretch" -> to'g'ridan-to'g'ri kvadratga cho'ziladi
                  "crop"    -> eski (HF standart) xatti-harakat

Pooling (DINOv2 uchun):
    "cls"      -> faqat CLS token (global tavsif)
    "cls+mean" -> CLS + patch tokenlar o'rtachasi
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .features import to_rgb_array


def _kind(model_name: str) -> str:
    n = model_name.lower()
    if "siglip" in n:
        return "siglip"
    if "clip" in n:
        return "clip"
    return "dino"


class _Backbone:
    """Bitta HF modeli + uning preprocessing'i."""

    def __init__(self, model_name, device, image_size, resize_mode, pooling):
        import torch
        from transformers import AutoConfig, AutoImageProcessor
        from transformers.utils import logging as hf_logging

        # "UNEXPECTED key" hisobotlari va progress-bar — foydalanuvchi uchun shovqin
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.torch = torch
        self.name = model_name
        self.kind = _kind(model_name)
        self.device = device
        self.resize_mode = resize_mode
        self.pooling = pooling

        proc = AutoImageProcessor.from_pretrained(model_name)
        self.mean = np.array(getattr(proc, "image_mean", [0.485, 0.456, 0.406]), np.float32)
        self.std = np.array(getattr(proc, "image_std", [0.229, 0.224, 0.225]), np.float32)
        self.image_size = int(image_size)
        if self.kind == "siglip":
            # SigLIP pozitsion embeddinglari qat'iy o'lchamga bog'langan
            cfg = AutoConfig.from_pretrained(model_name)
            self.image_size = int(getattr(getattr(cfg, "vision_config", cfg), "image_size", 224))

        if self.kind == "clip":
            from transformers import CLIPVisionModelWithProjection
            model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        elif self.kind == "siglip":
            # faqat vizual qism (matn enkoderi yuklanmaydi — xotira va vaqt tejaladi)
            try:
                from transformers import SiglipVisionModel
                model = SiglipVisionModel.from_pretrained(model_name)
            except Exception:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name).vision_model
        else:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
        self.dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.model = model.to(device=device, dtype=self.dtype).eval()

    def prep(self, img) -> np.ndarray:
        """Rasm -> (3, S, S) float32 normallashgan massiv (CPU yo'li, testlar uchun)."""
        x = self.prep_uint8(img).astype(np.float32) / 255.0
        return ((x - self.mean) / self.std).transpose(2, 0, 1)

    def prep_uint8(self, img) -> np.ndarray:
        """Rasm -> (S, S, 3) uint8 (faqat geometriya; normallashtirish GPU'da).

        PIL BICUBIC ataylab ishlatiladi: barcha benchmarklar aynan shu
        preprocessing bilan o'lchangan (OpenCV interpolatsiyasi kichik crop'larda
        embeddingni sezilarli o'zgartiradi: cos 0.95 gacha).
        """
        S = self.image_size
        pil = Image.fromarray(to_rgb_array(img))
        w, h = pil.size
        if self.resize_mode == "stretch":
            pil = pil.resize((S, S), Image.BICUBIC)
        elif self.resize_mode == "crop":
            short = 256 * S // 224
            if w < h:
                nw, nh = short, max(short, int(round(h * short / w)))
            else:
                nh, nw = short, max(short, int(round(w * short / h)))
            pil = pil.resize((nw, nh), Image.BICUBIC)
            l, t = (nw - S) // 2, (nh - S) // 2
            pil = pil.crop((l, t, l + S, t + S))
        else:  # pad (letterbox) — nisbat saqlanadi
            scale = S / max(w, h)
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            pil = pil.resize((nw, nh), Image.BICUBIC)
            fill = tuple(int(c * 255) for c in self.mean)  # normallashgandan keyin ~0
            canvas = Image.new("RGB", (S, S), fill)
            canvas.paste(pil, ((S - nw) // 2, (S - nh) // 2))
            pil = canvas
        return np.asarray(pil, dtype=np.uint8)

    def forward(self, x):
        torch = self.torch
        out = self.model(pixel_values=x)
        if self.kind == "clip":
            return out.image_embeds
        if self.kind == "siglip":
            return out.pooler_output
        hs = out.last_hidden_state
        cls = hs[:, 0]
        if self.pooling == "cls":
            return cls
        mean = hs[:, 1:].mean(dim=1)
        if self.pooling == "mean":
            return mean
        nf = torch.nn.functional
        return torch.cat([nf.normalize(cls, dim=1), nf.normalize(mean, dim=1)], dim=1)


class Embedder:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        batch_size: int = 64,
        image_size: int = 224,
        resize_mode: str = "pad",
        pooling: str = "cls",
        tta_flip: bool = False,
        weights: list[float] | None = None,
    ):
        import torch

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tta_flip = tta_flip
        self.model_name = model_name
        names = [m.strip() for m in str(model_name).split("+") if m.strip()]

        def per_model(v, what):
            # bitta qiymat -> hammasiga; ro'yxat -> har modelga alohida
            if isinstance(v, (list, tuple)):
                if len(v) != len(names):
                    raise ValueError(f"embedder.{what} soni modellar soniga teng bo'lishi kerak")
                return list(v)
            return [v] * len(names)

        self.backbones = [_Backbone(n, device, image_size, rm, pl) for n, rm, pl in
                          zip(names, per_model(resize_mode, "resize_mode"),
                              per_model(pooling, "pooling"))]
        w = weights or [1.0] * len(names)
        if len(w) != len(names):
            raise ValueError("embedder.weights soni modellar soniga teng bo'lishi kerak")
        self.weights = [float(x) for x in w]
        self._dim: int | None = None

    # Sweep/benchmark uchun: barcha backbone'larda preprocessing'ni o'zgartirish
    @property
    def resize_mode(self) -> str:
        return self.backbones[0].resize_mode

    @resize_mode.setter
    def resize_mode(self, v) -> None:
        vals = v if isinstance(v, (list, tuple)) else [v] * len(self.backbones)
        for b, x in zip(self.backbones, vals):
            b.resize_mode = x

    @property
    def pooling(self) -> str:
        return self.backbones[0].pooling

    @pooling.setter
    def pooling(self, v) -> None:
        vals = v if isinstance(v, (list, tuple)) else [v] * len(self.backbones)
        for b, x in zip(self.backbones, vals):
            b.pooling = x
        self._dim = None

    @property
    def dim(self) -> int:
        """Embedding o'lchami (masalan DINOv2-base CLS uchun 768)."""
        if self._dim is None:
            dummy = np.full((64, 64, 3), 127, dtype=np.uint8)
            self._dim = int(self.embed([dummy]).shape[1])
        return self._dim

    def _embed_one(self, bb: _Backbone, images: list) -> np.ndarray:
        torch = self.torch
        nf = torch.nn.functional
        mean = torch.tensor(bb.mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(bb.std, device=self.device).view(1, 3, 1, 1)
        out = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            # CPU: faqat uint8 resize (parallel — OpenCV GIL'ni bo'shatadi)
            arrs = list(self._pool().map(bb.prep_uint8, chunk))
            x = torch.from_numpy(np.stack(arrs)).to(self.device, non_blocking=True)
            # GPU: float, normallashtirish, NCHW
            x = ((x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std).to(bb.dtype)
            with torch.inference_mode():
                f = nf.normalize(bb.forward(x).float(), dim=1)
                if self.tta_flip:
                    f = nf.normalize(
                        f + nf.normalize(bb.forward(torch.flip(x, dims=[3])).float(), dim=1),
                        dim=1)
            out.append(f.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def _pool(self):
        if getattr(self, "_executor", None) is None:
            import os
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))
        return self._executor

    def embed(self, images: list) -> np.ndarray:
        """PIL yoki numpy(RGB) rasmlar ro'yxati -> (N, dim) L2-normallashgan vektorlar."""
        if not images:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        parts = []
        for bb, w in zip(self.backbones, self.weights):
            parts.append(np.sqrt(w) * self._embed_one(bb, images))
        v = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        return (v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)).astype(np.float32)

    def embed_crops(self, image: np.ndarray, boxes) -> np.ndarray:
        """To'liq rasm + Box ro'yxatidan crop'larni kesib embed qiladi."""
        return self.embed(crop_boxes(image, boxes))


def crop_boxes(image: np.ndarray, boxes) -> list[np.ndarray]:
    """Box'lar bo'yicha crop'lar (rasm chegarasiga qisilgan). Bo'sh crop -> kulrang."""
    crops = []
    h, w = image.shape[:2]
    for b in boxes:
        x1 = max(0, int(b.x1)); y1 = max(0, int(b.y1))
        x2 = min(w, int(round(b.x2))); y2 = min(h, int(round(b.y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            crops.append(np.full((8, 8, 3), 127, dtype=np.uint8))
        else:
            crops.append(image[y1:y2, x1:x2])
    return crops


def build_embedder(cfg, device: str) -> Embedder:
    return Embedder(
        model_name=cfg.get("embedder.model", "facebook/dinov2-base"),
        device=device,
        batch_size=cfg.get("embedder.batch_size", 64),
        image_size=cfg.get("embedder.image_size", 224),
        resize_mode=cfg.get("embedder.resize_mode", "pad"),
        pooling=cfg.get("embedder.pooling", "cls"),
        tta_flip=cfg.get("embedder.tta_flip", False),
        weights=cfg.get("embedder.weights", None),
    )
