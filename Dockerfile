# CiteShield API (FastAPI + SSE, browser UI) on CPU.
#
# The image holds the code and the validated corpus (data/processed). Not baked in:
#   - the Qdrant index (data/index, built locally by citeshield-index): mount it read-write
#     (embedded Qdrant takes a lock file in it);
#   - model weights (bge-m3, bge-reranker-v2-m3, DeBERTa NLI, about 7.7 GB): downloaded into
#     /models on first start; mount a named volume there to keep them;
#   - the LLM: a local Ollama server (CITESHIELD_LLM__PROVIDER=ollama) or Anthropic
#     (ANTHROPIC_API_KEY, paid). The application default is Anthropic; compose.yaml and the
#     test scripts set Ollama.
#
#   docker build -t citeshield:local .
#   docker compose up            (see compose.yaml and docs/development.md#docker)
#
# Without the index the server still starts: GET /health answers 200 and reports
# status "error" with the reason, GET /ready answers 503 (scripts/docker_smoke.py checks this).
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH" \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

WORKDIR /app

# Dependencies first (cached layer). On Linux the lockfile resolves torch to CPU wheels. The
# uv download cache is a build cache mount, so it never ends up in an image layer.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml --no-install-project

# The project is installed editable on purpose: Settings resolves data/ relative to
# src/citeshield, i.e. /app/data.
COPY src ./src
COPY data/corpus_manifest.yaml ./data/corpus_manifest.yaml
COPY data/processed ./data/processed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ml

# Non-root. It writes only to the model cache and the (mounted) index directory.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /models /app/data/index \
    && chown app:app /models /app/data/index
USER app

EXPOSE 8000
# Readiness: /ready returns 200 once the encoder, reranker, NLI model and index are loaded
# (the first start downloads the models). Liveness is GET /health, which always answers 200
# and reports loading / ok / error with the reason.
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/ready' % os.environ.get('UVICORN_PORT', '8000'), timeout=4)"]
# uvicorn is PID 1 and handles SIGTERM itself; host and port come from UVICORN_HOST/UVICORN_PORT.
CMD ["uvicorn", "citeshield.api.main:app"]
