# CiteShield API (FastAPI + SSE) on CPU.
#
# The image holds the code and the validated corpus (data/processed). Not baked in:
#   - the Qdrant index (data/index, built locally by citeshield-index): mount it;
#   - model weights (bge-m3, bge-reranker-v2-m3, DeBERTa NLI): downloaded into /models on
#     first start; mount a volume there to keep them;
#   - the LLM: Anthropic (ANTHROPIC_API_KEY) or a local Ollama server
#     (CITESHIELD_LLM__PROVIDER=ollama, CITESHIELD_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434).
#
#   docker build -t citeshield .
#   docker run --rm -p 8000:8000 -v ./data/index:/app/data/index -v citeshield-models:/models \
#     --env-file .env citeshield
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer). On Linux the lockfile resolves torch to CPU wheels.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra ml --no-install-project

COPY src ./src
COPY data/corpus_manifest.yaml ./data/corpus_manifest.yaml
COPY data/processed ./data/processed
RUN uv sync --locked --no-dev --extra ml

RUN useradd --create-home --uid 1000 app && mkdir -p /models /app/data/index \
    && chown -R app /models /app/data
USER app

EXPOSE 8000
# /health answers once the encoder, reranker and NLI model are loaded (first start downloads them).
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["uv", "run", "--no-sync", "uvicorn", "citeshield.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
