# Build ai/Dockerfile.base as rasta-ai-base:1 before building the AI service.
ARG BASE=rasta-ai-base:1
FROM ${BASE}
WORKDIR /app
COPY config.yaml .
COPY src/ src/
COPY ai/rasta_ai/ ai/rasta_ai/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/app/src:/app/ai
EXPOSE 8012
CMD ["python", "-m", "rasta_ai", "--host", "0.0.0.0"]
