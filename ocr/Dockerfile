# Rasta OCR — mahsulot yorlig'idagi yozuvni o'qiydigan alohida xizmat.
# Ixtiyoriy: `docker compose --profile ocr up -d --build` bilan yoqiladi.
# GPU: PaddlePaddle CUDA 12.6 g'ildiragi (Ampere va yangi kartalar, masalan A2) o'rnatiladi; qaysi
# qurilmada ishlashini RASTA_OCR_DEVICE belgilaydi (compose'da gpu:0). GPU ko'rinmasa xizmat o'zi CPU'ga
# o'tadi — modellar va natijalar bir xil. Faqat CPU image uchun:
#   docker build -f ocr/Dockerfile --build-arg PADDLE_WHEEL=paddlepaddle==3.3.1 -t rasta-ai-ocr .
# 3.11: ishlab chiquvchi kompyuteridagi Docker'da toza python:3.10-slim (3.10.22) oddiy kodda ham
# ~20% holatda ishga tushishda yiqildi, 3.11-slim esa 0/25 — tafsiloti BENCHMARK.md 11f da.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/ocr

# OpenCV va PaddlePaddle uchun tizim kutubxonalari
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# requirements.txt dagi paddlepaddle (CPU) qatori lokal muhit uchun; image'da o'rniga PADDLE_WHEEL o'rnatiladi.
# Rasmiy Paddle CDN'idagi g'ildirak (3.3.1, cp311, CUDA 12.6); CUDA kutubxonalari (cuDNN, cuBLAS...) PyPI'dan keladi.
ARG PADDLE_WHEEL=https://paddle-whl.cdn.bcebos.com/stable/cu126/paddlepaddle-gpu/paddlepaddle_gpu-3.3.1-cp311-cp311-linux_x86_64.whl
COPY ocr/requirements.txt ocr/requirements.txt
RUN pip install --upgrade pip \
 && grep -v '^paddlepaddle==' ocr/requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt "${PADDLE_WHEEL}"

COPY ocr/rasta_ocr/ ocr/rasta_ocr/
# Modellar image ichiga yuklanadi — konteyner internetsiz tarmoqda ishlaydi (qurish mashinasida GPU
# bo'lmasligi mumkin, shuning uchun bu yerda CPU)
RUN RASTA_OCR_DEVICE=cpu python -c "from rasta_ocr.service import PaddleReader; PaddleReader()"
# GPU xotirasi kerak bo'lgancha o'sadi (standart — kartaning yarmini darrov band qiladi; AI xizmati ham shu kartada)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_allocator_strategy=auto_growth

EXPOSE 8013
CMD ["python", "-m", "rasta_ocr", "--host", "0.0.0.0"]
