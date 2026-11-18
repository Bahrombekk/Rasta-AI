# Build Dockerfile.base as rasta-ai-base:1 before building the AI service.
ARG BASE=rasta-ai-base:1
FROM ${BASE}
WORKDIR /app
COPY config.yaml .
COPY src/ src/
COPY scripts/_bootstrap.py scripts/serve_ai.py scripts/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8012
CMD ["python", "scripts/serve_ai.py", "--host", "0.0.0.0"]
