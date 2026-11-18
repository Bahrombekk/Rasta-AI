# Rasta AI — Minimum server konfiguratsiyasi

Loyiha quyidagilarni bitta serverda ishga tushiradi (docker-compose):

| Komponent | Nima | Resurs talabi |
|---|---|---|
| **Nginx frontend** | Sayt va API proksi (standart port 8090) | yengil |
| **FastAPI backend** | Hisoblar, vektor katalogi, tahlil natijalari | RAM + CPU |
| **Qdrant (ichki/lokal rejim)** | SKU vektor bazasi — `data/qdrant` volume | RAM + disk |
| **Detektor ansambli** | yolo11-s (SKU-110K) + yolo11m (COCO butilka) | GPU/CPU |
| **Embedding ansambli** | DINOv2-base + SigLIP2-base | GPU/CPU (eng og'iri) |
| **FastAPI AI xizmati** | Alohida konteynerda deteksiya va embedding hisoblash | GPU/CPU |

> Eslatma: Qdrant **alohida server emas** — `qdrant-client` lokal (fayl) rejimida ishlaydi, volume'da saqlanadi. Alohida ma'lumotlar bazasi serveri shart emas.

---

## 1. Minimum — CPU-only (demo / kam yuklama)

GPU'siz ham ishlaydi (`config.yaml` da `device: auto` → GPU yo'q bo'lsa CPU). Ansamblda 2 ta embedding modeli borligi uchun sekinroq.

| Resurs | Minimum |
|---|---|
| CPU | 4 yadro (x86-64, AVX2) |
| RAM | **8 GB** |
| Disk | **30 GB** SSD |
| GPU | shart emas |
| OS | Ubuntu 22.04 / Debian 12 (yoki Docker bo'lgan har qanday Linux) |

**Tezlik:** ~5–15 soniya / rasm (ansambl, CPU). Bitta foydalanuvchi yoki demo uchun yetarli.
**Mos:** namoyish, pilot, kuniga bir necha yuz rasm.

---

## 2. Tavsiya etilgan minimum — GPU bilan (haqiqiy ishlatish)

| Resurs | Minimum | Qulay |
|---|---|---|
| GPU | **NVIDIA 6 GB VRAM** (T4, RTX 3060, A10) | 8–12 GB VRAM |
| CPU | 4 yadro | 8 yadro |
| RAM | **16 GB** | 32 GB |
| Disk | **40 GB** SSD (NVMe) | 80 GB+ |
| OS | Ubuntu 22.04 + NVIDIA driver 535+ + CUDA 12.x + nvidia-container-toolkit |

**Tezlik:** ~0.3–1 soniya / rasm (GPU, batch bilan).
**Throughput:** bitta T4/A10 ~ kuniga 100k–500k rasm (yuklamaga qarab).

---

## 3. VRAM / RAM taqsimoti (nega shuncha)

**VRAM (GPU xotira), FP32 da taxminan:**
- SigLIP2-base: ~1.5 GB (eng kattasi)
- DINOv2-base (86M): ~0.4 GB
- yolo11m + yolo11-s: ~0.5 GB
- Aktivatsiyalar (batch 64 @ 224px, detektsiya @ 1280px): ~1.5–2.5 GB
- **Jami peak: ~4–5 GB** → **6 GB minimum**, 8 GB bemalol

> TTA/tiling (`config.yaml` → `tta.enabled: true`) yoqilsa VRAM va vaqt 2–4× oshadi — kuchliroq GPU kerak bo'ladi.

**RAM (tizim xotira):**
- PyTorch + CUDA runtime: ~2–3 GB
- Modellar (CPU kopiyasi, yuklashda): ~2 GB
- Qdrant lokal indeks (10k SKU × bir necha vektor): ~0.5–1 GB
- FastAPI + rasm ishlash: ~1–2 GB
- **Jami: ~8 GB minimum**

**Disk:**
- Docker image (torch-cuda bilan): ~8–10 GB
- HF kesh (DINOv2 + SigLIP2 modellari): ~1.5 GB
- Model og'irliklari (`models/weights/`): ~0.1 GB
- Qdrant volume + katalog rasmlari: SKU soniga qarab o'sadi
- OS + zaxira: ~10 GB
- **Jami: ~30–40 GB**

---

## 4. Bulutli (cloud) variantlar — taxminiy

| Provayder | Minimal mos tarif | Izoh |
|---|---|---|
| **GPU** | T4 16GB yoki A10 — AWS g4dn.xlarge, GCP n1+T4, yoki istalgan GPU VPS | eng yaxshi narx/tezlik |
| **CPU-only VPS** | 4 vCPU / 8–16 GB RAM (Hetzner CPX31, DigitalOcean) | demo/pilot uchun arzon |
| **O'z serveringiz** | RTX 3060/4060 (12GB) bo'lgan oddiy PC | ishlab chiqish + kichik ishlatish |

> Loyiha dastlab RTX 5090 (32GB) da ishlab chiqilgan, lekin **joriy MVP uchun 6–8 GB VRAM yetarli** — 5090 ortiqcha.

---

## 5. Kelajakda (hisobotdagi to'liq tizim) talablari oshadi

Agar narx yorlig'i OCR / VLM (Qwen3-VL, PaddleOCR-VL) qo'shilsa:
- VLM-8B: **~16–24 GB VRAM**
- VLM-32B (FP8): **~40–80 GB VRAM** (A100/H100)

Bunda VLM'ni alohida GPU serverga (yoki alohida konteyner) ajratish tavsiya etiladi — CV qatlami (joriy MVP) va VLM qatlami turli mashinalarda. Hozircha bu shart emas.

---

## 6. Tez ishga tushirish (Docker)

```bash
# NVIDIA GPU bo'lgan serverda (nvidia-container-toolkit o'rnatilgan bo'lishi kerak)
docker build -f Dockerfile.base -t rasta-ai-base:1 .
docker compose up -d --build        # -> http://localhost:8090
docker compose exec backend python scripts/manage_users.py superadmin --role super_admin
docker compose logs -f frontend backend ai
```

`.env`, AI tokeni, model og‘irliklari va HTTPS sozlamalari uchun [README](README.md#dockerda-ishga-tushirish) bo‘limiga qarang.

CPU-only serverda: `docker-compose.yml` dagi `gpus: all` qatorini olib tashlang (yoki izohga oling), qolgani o'zgarmaydi — `device: auto` avtomatik CPU'ga o'tadi.

---

## Qisqa javob

- **Eng minimum (demo):** 4 vCPU, 8 GB RAM, 30 GB SSD, GPU'siz.
- **Tavsiya (ishlatish):** NVIDIA 6–8 GB VRAM (T4/A10), 16 GB RAM, 40 GB NVMe SSD, Ubuntu 22.04 + CUDA 12.
