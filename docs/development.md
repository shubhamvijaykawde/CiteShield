# Development

Tests, CI, Docker and observability. Architecture is in [architecture.md](architecture.md),
evaluation in [evaluation.md](evaluation.md).

## Running the server without the launcher

`scripts/start-demo.ps1` sets the local provider and memory-saving options for its own
process, so the demo never calls a paid API, whatever `.env` contains. The minimal manual
equivalent in PowerShell is:

```powershell
$env:CITESHIELD_LLM__PROVIDER = "ollama"; $env:CITESHIELD_LLM__OLLAMA_NUM_CTX = "24576"
$env:CITESHIELD_RETRIEVER__EMBED_DEVICE = "cpu"; $env:CITESHIELD_RETRIEVER__RERANK_DEVICE = "cpu"
py -m uv run uvicorn citeshield.api.main:app --host 127.0.0.1 --port 8000
```

Without `CITESHIELD_LLM__PROVIDER=ollama` the server uses the Anthropic default. The
launcher additionally sets `OLLAMA_NUM_PREDICT=8192`, a bfloat16 embedder and smaller batch
sizes; see the script.

## Tests and static checks

From PowerShell in the repository folder (`py -m uv` works without `uv` on `PATH`):

```powershell
py -m uv sync --extra ml
py -m uv run pytest
py -m uv run ruff check .
py -m uv run ruff format --check .
py -m uv run mypy src
```

`uv sync` installs the `dev` group (pytest, ruff, mypy) by default. `--extra ml` adds
torch (CUDA wheels on Windows, CPU wheels on Linux), transformers and huggingface-hub.

`pytest` runs the deterministic suite. It needs no models, keys or index: LLMs are scripted
and the NLI model is replaced by stand-ins. The default configuration deselects two opt-in
suites:

```powershell
py -m uv run pytest -m nli tests/nli     # the real NLI model; needs its weights and the corpus
py -m uv run pytest -m live tests/live   # a running Ollama with qwen3:4b and the built index
```

The release result (0.1.0) is in [RELEASE.md](../RELEASE.md#tests-and-deterministic-evaluation).
Without the `observability` extra, the one test that uses the real MLflow is skipped.

## CI

`.github/workflows/ci.yml` runs on pushes to `main`, on pull requests and on manual
dispatch, without secrets, model downloads or LLM calls:

- ruff, format and mypy;
- a **blocking security and verification gate**: unsupported and partly supported claims,
  invalid, fabricated and untrusted citations, quote mismatches, the refusal policy,
  poisoned retrieval and tool output (delivery, adoption, attribution), nonexistent
  provision references, and the web API;
- the full suite;
- a Docker image build (not pushed).

These tests use scripted LLMs and NLI stand-ins. They check the pipeline's rules, not a
model's faithfulness on new questions.

The model-quality gate (`eval-gate.yml`, Claude, faithfulness and citation regressions
against a baseline) is manual-only and has never run; see
[evaluation.md](evaluation.md#model-quality-gate).

## Docker (CPU)

```powershell
docker build -t citeshield .
docker run --rm -p 127.0.0.1:8000:8000 -v ./data/index:/app/data/index -v citeshield-models:/models --env-file .env citeshield
```

The image contains the code and the validated corpus. The index (`data/index`, built with
`citeshield-index`) and the model cache are mounted. The health check uses `GET /ready`.

The container does **not** force the local provider the way `start-demo.ps1` does. Without
an override it uses the Anthropic default, which needs `ANTHROPIC_API_KEY` and makes paid
calls. For Ollama on the host, set these in `.env` (copy `.env.example`):

```
CITESHIELD_LLM__PROVIDER=ollama
CITESHIELD_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434
```

CI builds the image; it has not been run end to end on the development laptop.

## Observability (optional)

```powershell
py -m uv sync --extra ml --extra observability
```

- **MLflow:** add `--mlflow` to `citeshield-eval`, `citeshield-eval-refusal` or
  `citeshield-eval-adversarial --score`. Each run is logged to `./mlruns` (gitignored) or to
  `MLFLOW_TRACKING_URI`. It records the run identity (models, prompt versions, config
  hashes, corpus, golden set, git commit), every numeric metric with counts and
  denominators for rates, and the metadata as an artifact. No cost is logged for Ollama,
  which does not bill, and token counts come only from the provider. The default store is
  MLflow 3's deprecated file store, which the adapter enables with
  `MLFLOW_ALLOW_FILE_STORE=true` because mlflow-skinny has no SQL backend; set a server URI
  to avoid it.
- **Langfuse:** set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`
  (cloud or self-hosted). Each `/ask` or `citeshield-ask` request then becomes a trace
  containing:
  - an agent observation;
  - the guard;
  - one retriever/tool observation per call;
  - one generation per LLM call, with only provider-reported token counts;
  - warning events for rejected drafts and failed verification;
  - the outcome (status, refusal reason, citations, support ratio).

  Without keys no client is created and nothing is sent; `/health` reports
  `"tracing": "off"`, and the demo and tests run unchanged. `CITESHIELD_TRACING=off`
  disables it. The adapter is unit-tested against a fake client; it has not been exercised
  against a live Langfuse server.

## Rebuilding the corpus

`py -m uv run citeshield-ingest` re-downloads the pinned sources
(`data/corpus_manifest.yaml`) from the Publications Office Cellar API and rebuilds
`data/processed/chunks.jsonl`. It only matters if the corpus changes; afterwards rebuild the
index with `py -m uv run citeshield-index`. The retriever refuses an index built for a
different corpus hash.
