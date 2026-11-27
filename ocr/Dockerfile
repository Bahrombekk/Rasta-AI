# Rasta OCR — mahsulot yorlig'idagi yozuvni o'qiydigan alohida xizmat.
# Ixtiyoriy: `docker compose --profile ocr up -d --build` bilan yoqiladi.
# GPU: PaddlePaddle CUDA 12.6 g'ildiragi (Ampere va yangi kartalar, masalan A2) o'rnatiladi; qaysi
# qurilmada ishlashini RASTA_OCR_DEVICE belgilaydi (compose'da gpu:0). GPU ko'rinmasa xizmat o'zi CPU'ga
# o'tadi — modellar va natijalar bir xil. Faqat CPU image uchun:
#   docker build -f ocr/Dockerfile --build-arg PADDLE_WHEEL=paddlepaddle==3.3.1 -t rasta-ai-ocr .
# 3.11: ishlab chiquvchi kompyuteridagi Docker'da toza python:3.10-slim (3.10.22) oddiy kodda ham
# ~20% holatda ishga tushishda yiqildi, 3.11-slim esa 0/25 — tafsiloti BENCHMARK.md 11f da.
# 1-bosqich: modellarni yuklash. GPU'li PaddlePaddle `import` paytidayoq NVIDIA drayverining libcuda.so.1
# kutubxonasini talab qiladi, u esa faqat konteyner GPU bilan ishga tushganda ulanadi — qurish paytida yo'q.
# Shuning uchun modellar CPU PaddlePaddle bilan yuklanadi va tayyor fayllar oxirgi image'ga ko'chiriladi
# (modellar freymvorkka bog'liq emas). Nomlar ocr/rasta_ocr/service.py dagi DET_MODEL va REC_MODEL bilan bir xil
# bo'lishi kerak (tests/test_ocr.py tekshiradi).
FROM python:3.11-slim AS models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY ocr/requirements.txt ocr/requirements.txt
RUN pip install --upgrade pip && pip install -r ocr/requirements.txt
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv5_mobile_det', text_recognition_model_name='PP-OCRv6_medium_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', enable_mkldnn=False)"

# 2-bosqich: xizmatning o'zi
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

# Modellar image ichiga ko'chiriladi — konteyner internetsiz tarmoqda ishlaydi (1-bosqichga qarang)
COPY --from=models /root/.paddlex /root/.paddlex
COPY ocr/rasta_ocr/ ocr/rasta_ocr/
# GPU xotirasi kerak bo'lgancha o'sadi (standart — kartaning yarmini darrov band qiladi; AI xizmati ham shu kartada)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_allocator_strategy=auto_growth

EXPOSE 8013
CMD ["python", "-m", "rasta_ocr", "--host", "0.0.0.0"]
