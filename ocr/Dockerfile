# Rasta OCR — mahsulot yorlig'idagi yozuvni o'qiydigan alohida xizmat (CPU).
# Ixtiyoriy: `docker compose --profile ocr up -d --build` bilan yoqiladi.
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
COPY ocr/requirements.txt ocr/requirements.txt
RUN pip install --upgrade pip && pip install -r ocr/requirements.txt

COPY ocr/rasta_ocr/ ocr/rasta_ocr/
# Modellar image ichiga yuklanadi — konteyner internetsiz tarmoqda ishlaydi
RUN python -c "from rasta_ocr.service import PaddleReader; PaddleReader()"
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

EXPOSE 8013
CMD ["python", "-m", "rasta_ocr", "--host", "0.0.0.0"]
