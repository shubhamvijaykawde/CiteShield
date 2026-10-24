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

`.github/workflows/ci.yml` runs on every pull request, on pushes to `main` and on manual
dispatch, without secrets, paid APIs, model downloads or LLM calls. A newer push to the same
pull request cancels its older run. Two jobs:

**`lint-and-test`** (Python 3.12, `uv sync --locked --group dev`):

- ruff, format and mypy;
- a **blocking security and verification gate**: unsupported and partly supported claims,
  invalid, fabricated and untrusted citations, quote mismatches, the refusal policy,
  poisoned retrieval and tool output (delivery, adoption, attribution), nonexistent
  provision references, and the web API;
- the full deterministic suite.

**`docker-build`**:

- `docker compose config --quiet` (validates `compose.yaml`);
- builds the image with Buildx (GitHub Actions layer cache, `push: false`: nothing is
  published);
- runs the [container smoke test](#container-smoke-test-no-models-no-llm) against it.

A failure in any step fails the workflow. The tests use scripted LLMs and NLI stand-ins:
they check the pipeline's rules, not a model's faithfulness on new questions. There is no
deployment stage.

The model-quality gate (`eval-gate.yml`, Claude, faithfulness and citation regressions
against a baseline) is manual-only and has never run; see
[evaluation.md](evaluation.md#model-quality-gate). The answer evaluations and adversarial
reports run locally ([evaluation.md](evaluation.md#commands)), not in CI.

To make the checks block merging, protect `main` on GitHub (*Settings → Rules → Rulesets*,
or *Settings → Branches*) and add `lint-and-test` and `docker-build` as required status
checks. That is a repository setting; the workflow cannot set it.

## Docker (CPU)

The Docker path runs the same server as the native demo, on CPU, with the local Ollama
model. It is an alternative to `scripts/start-demo.ps1`, which stays the primary demo on
Windows.

| | native demo (`start-demo.ps1`) | Docker (`compose.yaml`, test scripts) |
|---|---|---|
| Python | `.venv` from `uv sync --extra ml` (CUDA torch on Windows) | image: Python 3.12, `uv sync --locked --no-dev --extra ml` (CPU torch) |
| index | `data\index` | `./data/index` mounted at `/app/data/index` (read-write: Qdrant takes a lock file) |
| Hugging Face models | `%USERPROFILE%\.cache\huggingface` | volume `citeshield-models` at `/models` (`HF_HOME`) |
| Ollama URL | `http://127.0.0.1:11434` | `http://host.docker.internal:11434` |
| provider | forced to `ollama` | forced to `ollama` by `compose.yaml` and the scripts; the image itself defaults to Anthropic |
| URL | http://127.0.0.1:8000 | http://127.0.0.1:8000 (published on loopback only) |

**In the image:** the code, the dependencies from `uv.lock` and the validated corpus
(`data/processed`, `data/corpus_manifest.yaml`). `.dockerignore` is an allowlist, so
`.env`, `.git`, `.venv`, `data/index`, `data/raw`, model caches, tests and evaluation output
never enter the build context. The server runs as user `app` (uid 1000), with uvicorn as
PID 1 on `0.0.0.0:$UVICORN_PORT` (default 8000) inside the container; `docker stop` shuts
it down cleanly. The image is about 2 GB, mostly CPU torch.

**Not in the image:**

- the retrieval index: build it natively once (`py -m uv run citeshield-index`) and mount it;
- the Hugging Face models (bge-m3, bge-reranker-v2-m3, DeBERTa NLI; the volume measured
  7.7 GB): downloaded into the `citeshield-models` volume on the first start;
- the LLM: Ollama with `qwen3:4b` on the host, or the opt-in container below.

**Health.**

- `GET /health` is the liveness check. It answers 200 whenever the process runs and
  reports `status` (`loading`, `ok` or `error` with the reason), the LLM backend, and
  whether Ollama is reachable and has the model.
- `GET /ready` is the readiness check: 200 once the encoder, reranker, NLI model and index
  are loaded, otherwise 503. The image's `HEALTHCHECK` uses it, with a 15-minute start
  period for the first model download. `docker ps` therefore shows `healthy` only when the
  server can answer, and `unhealthy` when the agent failed to load (for example without an
  index).

### Commands (PowerShell, Windows with Docker Desktop)

Prerequisites: Docker Desktop running (WSL 2 backend), Ollama running with
`ollama pull qwen3:4b`, and the index in `data\index`. The models need about 6 GB of memory
in Docker Desktop's VM (by default it gets half of the host's RAM).

```powershell
docker compose up --build -d              # build and start: http://127.0.0.1:8000
docker compose logs -f api                # startup and request logs (Ctrl+C stops following)
docker compose ps                         # STATUS becomes "healthy" when /ready is 200
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose down                       # stop and remove the container
```

Plain Docker, without Compose:

```powershell
docker build -t citeshield:local .
docker run --rm -p 127.0.0.1:8000:8000 -v "${PWD}\data\index:/app/data/index" -v citeshield-models:/models `
  -e CITESHIELD_LLM__PROVIDER=ollama -e CITESHIELD_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434 citeshield:local
```

Without `CITESHIELD_LLM__PROVIDER=ollama` the container uses the application default,
Anthropic, which needs `ANTHROPIC_API_KEY` and makes paid calls.

**Configuration.** `compose.yaml` sets the Ollama provider and the memory-saving settings
of `start-demo.ps1`. It also reads an optional `.env` (gitignored; for example `LANGFUSE_*`
keys); the values in `compose.yaml` take precedence. Two Compose-only variables come from
the shell or `.env`: `CITESHIELD_PORT` (host port, default 8000) and
`CITESHIELD_DOCKER_OLLAMA_URL` (default `http://host.docker.internal:11434`). Nothing
secret is in the image or in `compose.yaml`. `docker compose config` prints the values it
read from `.env`, so do not share its output.

**Ollama networking.** Ollama keeps its default address, `127.0.0.1:11434`. Docker Desktop
routes `host.docker.internal` to the host's loopback, so the container reaches Ollama
without Ollama being exposed to the network. On Linux Docker Engine,
`host.docker.internal` (mapped to `host-gateway`) is the bridge address, which an Ollama
bound to 127.0.0.1 does not answer. There, use the Ollama container instead of binding the
host's Ollama to `0.0.0.0`:

```powershell
$env:CITESHIELD_DOCKER_OLLAMA_URL = "http://ollama:11434"
docker compose --profile ollama up -d ollama
docker compose exec ollama ollama pull qwen3:4b   # about 2.5 GB, into the citeshield-ollama-models volume
docker compose --profile ollama up --build -d
```

That container runs on CPU (no GPU is configured), so answers are much slower, and it needs
about 4 GB more memory in the Docker VM. Its port is not published: only the `api`
container reaches it, over the Compose network.

### Container smoke test (no models, no LLM)

```powershell
py -m uv run python scripts/docker_smoke.py                                    # builds citeshield:smoke, then tests it
py -m uv run python scripts/docker_smoke.py --image citeshield:local --skip-build
```

The script starts one container from the image on a random loopback port. Ollama is pointed
at a closed port and **no index** is mounted. It then checks the running server:

- `/health` answers 200 with the documented fields;
- it reports `status: "error"` naming the missing index manifest, and an unreachable,
  unpaid Ollama backend;
- `/ready` and `POST /ask` answer 503;
- `/` serves the UI, and the OpenAPI schema lists `/ask`, `/health` and `/ready`;
- the server does not run as root, and the image contains no `.env`, `.git` or index.

On failure it prints the container logs. It removes only its own container. CI runs the
same script. It needs only Python 3 (standard library) and Docker.

This proves that the packaged application starts, serves its API and honestly reports that
it is not ready. It does not load models, read an index or answer a question.

### Real-model end-to-end test (local, opt-in)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\docker-e2e.ps1                       # builds citeshield:local
powershell -ExecutionPolicy Bypass -File scripts\docker-e2e.ps1 -SkipBuild -StrictAnswer
```

1. Checks Docker, Ollama on `127.0.0.1:11434` with `qwen3:4b`, the corpus (including GDPR
   Art. 33(1)) and `data\index`, with an actionable message for each.
2. Builds the image (unless `-SkipBuild`) and starts a container with the Compose settings,
   a **temporary copy** of the index and the `citeshield-models` volume. `data\index` is
   never written to, and a running native demo is not disturbed.
3. Waits for `/health` status `ok` and `/ready` 200 (`-ReadyTimeoutSec`, default 2400 s,
   because the first run downloads the models) and checks that the container reaches
   Ollama.
4. Asks *"Within how many hours must a controller notify a personal data breach to the
   supervisory authority under the GDPR?"* over `POST /ask`, reads the SSE stream and
   prints the stages.
5. Validates the `final` event. `answer.status` must be answered, partial or refused. A
   refusal must have a reason and no claims. Otherwise every claim and checklist item needs
   citations with a `chunk_id` and an 8–400 character quote, every cited chunk must be in
   `sources`, and the verification report must be present.
6. Reports answer quality separately: whether the answer cites GDPR Art. 33(1) and
   mentions 72 hours.
7. Removes its container and the index copy, also on failure (after printing the logs).
   It keeps the image and the model volume. `-KeepContainer` keeps the container.

The exit code separates the failure kinds:

| exit | meaning |
|---|---|
| 0 | transport and schema passed (the output says whether answer quality was met) |
| 1 | a prerequisite is missing; nothing was started |
| 2 | infrastructure or transport: build, start, readiness, Ollama unreachable from the container, HTTP, SSE, an `error` event |
| 3 | schema: the final payload is not a valid answer |
| 4 | model quality, only with `-StrictAnswer`: no answer citing Art. 33(1) |

Without `-StrictAnswer`, a refusal or an answer without that citation still exits 0, but
the result line says **transport and schema only, not evidence of a correct answer**.
qwen3:4b is nondeterministic, so one run says little about answer quality; the evaluations
in [evaluation.md](evaluation.md) measure it.

### What each check proves

| check | proves | does not prove |
|---|---|---|
| `docker build` (CI) | the Dockerfile and lockfile install on Linux | that the server starts |
| container smoke test (CI) | the image starts, serves the UI and API, reports liveness and an honest not-ready state, runs as non-root, contains no `.env` or index | model loading, retrieval, answers |
| deterministic suite and security gate (CI) | the pipeline's rules, with scripted LLMs and NLI stand-ins | a real model's faithfulness |
| `docker-e2e.ps1` (local) | the packaged app loads the real models and index, reaches Ollama, and returns a valid structured answer with citations over SSE | answer quality in general (one nondeterministic question) |
| answer evaluations ([evaluation.md](evaluation.md)) | measured quality on the question sets | anything beyond those single runs |

### Troubleshooting

- *failed to connect to the docker API ... dockerDesktopLinuxEngine*: Docker Desktop is not
  running. Start it and wait for "Engine running".
- `/health` reports `status: "error"` with `.../manifest.json`: no index is mounted. Build
  it natively (`py -m uv run citeshield-index`); Compose mounts `./data/index`.
- `/health` reports `llm.reachable: false`: the container cannot reach Ollama. Check that
  the Ollama app runs and that `CITESHIELD_DOCKER_OLLAMA_URL` is right (see Ollama
  networking).
- `status` stays `loading` on the first start: the models are downloading into
  `citeshield-models` (`docker compose logs -f api`).
- The container is killed, or the host swaps: raise Docker Desktop's memory limit
  (*Settings → Resources*, or `memory=` in `%USERPROFILE%\.wslconfig`), or stop the native
  demo while the container runs.
- Port 8000 is taken (for example by the native demo): `$env:CITESHIELD_PORT = "8001"`.

### Cleanup

```powershell
docker compose down                             # the container and its network
docker compose --profile ollama down            # also the opt-in Ollama container
docker ps -a --filter label=citeshield.test     # leftovers of interrupted test runs: docker rm -f <name>
docker image rm citeshield:local citeshield:smoke
docker volume rm citeshield-models citeshield-ollama-models   # deletes the downloaded models
```

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
