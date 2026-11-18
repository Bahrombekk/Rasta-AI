# Rasta AI — veb-ilova (GPU: NVIDIA, CUDA 12.8 — RTX 50xx/Blackwell ham)
#
# Birinchi marta (bir marta, ~20 daqiqa — PyTorch va modellar):
#   docker build -f Dockerfile.base -t rasta-ai-base:1 .
# Keyin (soniyalar):
#   docker compose up -d --build        ->  http://localhost:8090   (admin: /admin)
#
# Og'ir qism alohida image'da: Docker Desktop qurish keshini tozalasa ham PyTorch qayta
# yuklanmaydi. Modellar image ichida — konteyner internetsiz ishlaydi.
ARG BASE=rasta-ai-base:1
FROM ${BASE}

WORKDIR /app

# Detektor og'irliklari Compose orqali /app/models/weights ga ulanadi.
COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/
COPY tools/ tools/

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Katalog (Qdrant) va mahsulot rasmlari — volume'larda saqlanadi
VOLUME ["/app/data/qdrant", "/app/data/catalog_images"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import json, urllib.request; status = json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)); raise SystemExit(0 if status['loaded'] else 1)"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
