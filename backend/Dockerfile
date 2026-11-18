FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app/backend
WORKDIR /app
COPY backend/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && python -m pip install -r requirements.txt
COPY config.yaml .
COPY src/ src/
COPY backend/rasta_backend/ backend/rasta_backend/
COPY scripts/_bootstrap.py scripts/manage_users.py scripts/
EXPOSE 8000
CMD ["python", "-m", "rasta_backend", "--host", "0.0.0.0", "--port", "8000", "--api-only"]
