# CiteShield

A cited, injection-resistant agent that answers compliance questions over public EU legal
texts (GDPR, AI Act, DSA, NIS2, Data Act) and refuses when the evidence is thin.

> **Local demonstration project.** CiteShield runs on your own machine with a local model
> (Ollama `qwen3:4b`); **no public demo is hosted**. It gives experimental legal
> information, not legal advice or a compliance certification.

Version 0.1.0 is the final portfolio release; research is frozen. **[RELEASE.md](RELEASE.md)**
has the verified results, their definitions and denominators, the security findings and the
known limitations.

## What it does

You ask a question about one of five EU regulations, compare provisions, or request a
compliance checklist. CiteShield first researches the official texts with search tools,
then drafts an answer as structured JSON (`AgentAnswer`, a Pydantic model). In that answer
every claim carries a paragraph-level citation and a verbatim quote.

Before you see anything, a verifier checks each citation and quote against the canonical
corpus. An NLI model then checks that the cited text entails each claim, and each
coordinated clause in it. Anything unsupported is removed, and the answer is marked
partial. When nothing survives, or the question is out of scope, you get a typed refusal
with a reason.

| capability | status |
|---|---|
| Ingestion with a data contract (identity, version, article/paragraph/point, provenance) and quarantine | implemented, tested |
| Hybrid dense+sparse retrieval with a reranker, stable chunk ids | implemented, evaluated |
| LangGraph agent: answer / compare / checklist modes, structured output, refusal policy | implemented, evaluated on 14 questions (single runs) |
| Claim-level verification (citation, verbatim quote, NLI entailment per claim and clause) | implemented; thresholds provisional |
| Local browser UI + FastAPI streaming API + CLI | implemented, tested offline; live smoke test in RELEASE.md |
| Deterministic evaluation: answer metrics, over-refusal, adversarial scoring | implemented |
| Adversarial evaluation (5 poisoned-content attacks) | **partial**: preliminary, ablations incomplete |
| LLM judge (qwen3:4b) | implemented, **failed calibration**, informational only |
| MLflow run tracking, Langfuse tracing | implemented, optional (off by default) |
| Docker image | implemented, built in CI; not run end to end locally |
| CI: lint, types, tests, security gate, image build | implemented, no API keys needed |
| Claude-based quality gate | workflow present, manual only, inactive (needs a key and a baseline) |
| Public hosted demo | not implemented (out of scope) |

Detailed reports: [over-refusal](eval/results/milestone7_over_refusal.md),
[adversarial evaluation and ablations](eval/results/milestone7_adversarial.md),
[judge calibration](eval/results/judge_calibration.md),
[Milestone 6 closeout](eval/results/milestone6_closeout.md).

## Run the demo on Windows

Requirements:

- Windows with Python 3.12 and the `py` launcher;
- [Ollama](https://ollama.com) with `qwen3:4b` pulled;
- about 8 GB of free RAM (an NVIDIA GPU helps; the retrieval models run on CPU);
- roughly 6 GB of disk for the Hugging Face models, downloaded on first start.

In PowerShell, from the repository folder:

```powershell
py -m pip install uv
py -m uv sync --extra ml
ollama pull qwen3:4b
py -m uv run citeshield-index
powershell -ExecutionPolicy Bypass -File scripts\start-demo.ps1
```

- `py -m uv sync --extra ml` installs everything (CUDA torch on Windows).
- `ollama pull qwen3:4b` is needed once.
- `citeshield-index` builds `data/index` from the committed corpus, once (a few minutes).
- `start-demo.ps1` checks the prerequisites and opens http://127.0.0.1:8000.

The launcher always selects the local Ollama backend (`CITESHIELD_LLM__PROVIDER=ollama`)
for that process, so the demo never calls a paid API, whatever `.env` contains. It binds the
server to 127.0.0.1 only. Without the script, the equivalent is:

```powershell
$env:CITESHIELD_LLM__PROVIDER = "ollama"; $env:CITESHIELD_LLM__OLLAMA_NUM_CTX = "24576"
$env:CITESHIELD_RETRIEVER__EMBED_DEVICE = "cpu"; $env:CITESHIELD_RETRIEVER__RERANK_DEVICE = "cpu"
py -m uv run uvicorn citeshield.api.main:app --host 127.0.0.1 --port 8000
```

### Demo walkthrough

1. **Wait for the status box** (top right) to say *Ready · local Ollama, qwen3:4b*. It shows
   *Loading* while the models load (one to two minutes). If startup fails, it shows the
   reason, for example a missing index, and so does `GET /health`.
2. **Ask a question.** Click an example or type one, choose a mode, optionally limit the
   regulations, and press *Ask*. Progress streams live: the scope check, tool calls with the
   provisions they returned, drafting and repairs. A local answer takes about 1–5 minutes.
3. **Read the answer.** It has:
   - a status badge (answered / partial / refused) and a summary;
   - each claim with its source link and verbatim quote;
   - checklist items in checklist mode;
   - qualifications;
   - how many claims passed verification.

   Click a source to see the full official paragraph with its CELEX number, version and an
   EUR-Lex link.
4. **Show a refusal.** The Irish corporate-tax example is refused before any research, with
   the reason *out_of_scope* and a plain-language explanation.

Example questions:

| mode | question | a good answer |
|---|---|---|
| answer | Within how many hours must a controller notify a personal data breach to the supervisory authority? | Without undue delay and, where feasible, within 72 hours, citing GDPR Art. 33(1) |
| answer | What conditions apply to consent, and can a data subject withdraw it? | Consent can be withdrawn at any time; withdrawal does not affect earlier lawful processing (GDPR Art. 7(3)) |
| compare | Compare the deadlines for reporting a personal data breach under the GDPR with significant incident reporting under NIS2. | GDPR 72 hours (Art. 33(1)) against NIS2 early warning in 24 h and notification in 72 h (Art. 23(4)) |
| checklist | Give me a checklist of the obligations of providers of high-risk AI systems. | Items from AI Act Art. 16, each with a citation |
| answer | What is the corporate income tax rate in Ireland? | Refused: outside the corpus |

qwen3:4b is small and varies from run to run. Expect some partial answers, and the
occasional unnecessary refusal (1 of 11 answerable questions in the last measured run).

## Alternative: Docker (CPU)

```bash
docker build -t citeshield .
docker run --rm -p 127.0.0.1:8000:8000 -v ./data/index:/app/data/index -v citeshield-models:/models --env-file .env citeshield
```

The image contains the code and the validated corpus. The index (`data/index`, built with
`citeshield-index`) and the model cache are mounted. For Ollama on the host, set
`CITESHIELD_LLM__PROVIDER=ollama` and
`CITESHIELD_LLM__OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`. The health
check uses `GET /ready`. CI builds the image; it has not been run end to end on the
development laptop.

## Tests, evaluation and CI

```powershell
py -m uv sync --group dev --extra ml
py -m uv run ruff check .; py -m uv run ruff format --check .; py -m uv run mypy src
py -m uv run pytest
py -m uv run citeshield-eval-refusal eval/runs/<run_id>
py -m uv run citeshield-eval-adversarial --score eval/runs/<adv_run_id>
```

- `pytest` runs the deterministic suite: no models, keys or index needed.
- `citeshield-eval-refusal` scores over-refusal, coverage, correct and under-refusal, and
  invalid outputs.
- `citeshield-eval-adversarial --score` reports attack delivery and success, with clean
  controls.
- Live evaluations, which call the LLM, are `citeshield-eval --subset local` and
  `citeshield-eval-adversarial --run-id ...`. Run records go to `eval/runs/` (gitignored).

CI (`.github/workflows/ci.yml`) runs on every pull request without secrets:

- ruff, format and mypy;
- a **blocking security and verification gate**: unsupported and partly supported claims,
  invalid, fabricated and untrusted citations, quote mismatches, the refusal policy,
  poisoned retrieval and tool output, and the web API;
- the full suite;
- a Docker image build.

These tests use scripted LLMs and NLI stand-ins. They check the pipeline's rules, not a
model's faithfulness on new questions.

The model-quality gate (`eval-gate.yml`, Claude, faithfulness and citation regressions
against a baseline) is manual-only. It fails explicitly when its API key or baseline is
missing.

## Observability (optional)

```powershell
py -m uv sync --extra ml --extra observability
```

- **MLflow:** add `--mlflow` to `citeshield-eval`, `citeshield-eval-refusal` or
  `citeshield-eval-adversarial --score`. Each run is logged to `./mlruns` (gitignored; MLflow
  3's deprecated file store, which the adapter enables with `MLFLOW_ALLOW_FILE_STORE=true`),
  or to `MLFLOW_TRACKING_URI`. It records the run identity (models, prompt versions, config
  hashes, corpus, golden set, git commit), every numeric metric, and the metadata as an
  artifact. No cost is logged for Ollama, which does not bill.
- **Langfuse:** set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`
  (cloud or self-hosted). Each API or CLI request then becomes a trace containing:
  - an agent observation;
  - the guard;
  - one retriever/tool observation per call;
  - one generation per LLM call, with only provider-reported token counts;
  - warning events for rejected drafts and failed verification;
  - the outcome (status, refusal reason, citations, support ratio).

  Without keys nothing is sent, and the demo and tests run unchanged.
  `CITESHIELD_TRACING=off` disables it. The adapter is unit-tested against a fake client; it
  has not been exercised against a live Langfuse server.

## Agent

`src/citeshield/agent/graph.py` defines the agent as a LangGraph state machine:

```
guard ──in scope──► agent ⇄ tools ──► draft ──► validate ──► final answer
  │                 (≤ 6 tool rounds)              │  ▲
  └──out of scope──► typed refusal                 └──┘ one repair, then drop unsupported claims
```

- **guard** (Claude Haiku 5.5, structured output): screens out questions about other laws,
  requests to assess the user's own situation, and attempts to change the assistant's
  instructions. These are refused before any retrieval runs. If the guard fails, the question
  goes through to research; the guard saves cost and is not the security boundary.
- **agent ⇄ tools** (Claude Sonnet 5.5): four read-only tools, `retrieve`, `get_article`,
  `compare_articles` and `generate_checklist`.
  - Each tool has a strict schema, and the arguments are validated again with Pydantic.
  - Calls to any other tool name are rejected and flagged.
  - Every chunk a tool returns goes into an evidence registry.
  - Source text is wrapped in `<source>` delimiters that the text itself can't close.
- **draft**: a separate call with a JSON-schema output (`output_config.format`) writes the
  `AgentAnswer`. Sonnet 5.5 can't be forced to call a tool, so the draft is a separate call
  whose JSON schema guarantees the format.
- **validate**: checks the Pydantic contract, and checks that every citation is a chunk some
  tool returned in this run. On errors it sends the draft back once with the error list. If
  the repair also fails, claims with unresolved citations are dropped and the answer becomes
  `partial`, or a refusal if nothing is left.
- **History is append-only.** Assistant turns go back to the API unchanged, including their
  thinking blocks.
- **Prompt caching and refusal fallback.** The system prompt and tool list are cached.
  Sonnet calls use the server-side refusal fallback (`fallbacks: "default"`).
- **Logged with each run**: prompt versions (content hashes), the agent, retriever and corpus
  hashes, token usage and estimated cost.

## Verification and refusal policy

Every drafted answer is verified before anyone sees it (`src/citeshield/verify/`):

1. **Citations** are checked against the canonical corpus and this run's evidence:
   `fabricated` (no such id), `version_mismatch` (same provision, other `@version`),
   `non_citable` (e.g. an untrusted source), `not_retrieved` (no tool returned it).
2. **Quotes** must match the canonical text after deterministic normalisation (NFKC,
   typographic quotes/dashes, whitespace, case). Near misses are reported, never accepted.
3. **Entailment**: a matching quote only proves the passage exists. Each claim must be
   entailed by its cited text according to a local NLI model
   (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, MIT, CPU by default).
   Verdicts: supported, unsupported, contradicted, conflicting, citation_invalid.
   A claim that joins obligations ("must send X and notify Y", or "A; B") must also have
   each clause entailed. Whole-sentence entailment alone once let a claim through that
   was only half supported (`VerifierConfig.check_conjuncts`). Other forms of partial
   support, such as an unsupported qualifier inside one clause, are not split out.
4. **Summary and caveats** are split into sentences; only sentences entailed by the verified
   claims or their sources are kept. Disclaimers are added by code.

Failures get one revision with per-claim reasons. After that `agent/policy.py` removes what
failed (answer becomes `partial`) or refuses with a specific reason
(`invalid_citations`, `unsupported_claims`, `conflicting_sources`, `insufficient_evidence`
when the best retrieval score is below 0.30). Verification can only lower a status. The
verifier is a required constructor argument: no agent can be built without it. The full
`VerificationReport` is returned in the final SSE event.

Thresholds come from `uv run citeshield-calibrate-verifier` on 60 hand-labelled pairs
(`eval/verification/claims.jsonl`, stratified calibration/held-out split); see
[eval/results/verifier_calibration.md](eval/results/verifier_calibration.md). The model
separates that set with a wide margin, so the data shows reliability on clear-cut claims but
does not pin the thresholds down; they stay provisional until the answer evals have been
labelled.

Opt-in test suites: `pytest -m nli tests/nli` (real model) and `pytest -m live tests/live`
(Ollama).

## Evaluation

The golden set (`eval/golden/questions.jsonl`) has 151 questions: lookups, definitions,
multi-paragraph, cross-regulation, comparison, checklist, temporal, and 30 (20%) that must be
refused (other laws, case law, enforcement facts, personal legal advice, injection, off-topic).
Each answerable item lists its expected citations at paragraph level and a short reference
answer. Items q061-q151 were drafted with every citation checked against the corpus and are
marked `"reviewed": false` until a human has read them. The reference answers of q001-q053
were also drafted from the text of their expected provisions and still need a human read.

```bash
uv run citeshield-eval --subset smoke          # 40 stratified questions (eval/golden/smoke.json)
uv run citeshield-eval --subset full --judge   # every item, plus the LLM judge
uv run citeshield-eval --rescore eval/runs/<run_id> [--judge]   # re-score, no agent calls
uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json   # compare with the baseline
uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json --write-baseline
uv run citeshield-label                        # hand-label the frozen sample (eval/labels)
uv run citeshield-label --coverage             # labelled vs. remaining, label problems
uv run citeshield-calibrate-judge              # judge the labelled items, kappa + report
```

A run writes `eval/runs/<run_id>/` (`meta.json`, `answers.jsonl`, `judge.jsonl`,
`metrics.json`, `report.md`; gitignored) and copies the report to
[eval/results/answers_latest.md](eval/results/answers_latest.md). Re-running the same
`--run-id` resumes where it stopped, and refuses if the agent configuration changed in
between. `--write-smoke` regenerates the smoke subset deterministically (proportional per
category, at least two items each); changing it requires a new baseline.

Metrics (`src/citeshield/evaluation/metrics.py`), all deterministic:

- **citation precision / recall**: cited chunks matching an expected paragraph, and
  expected paragraphs that were cited.
- **refusal precision / recall / F1**: a deliberate refusal is the positive class. A crash
  (no answer at all, e.g. a context overflow) is never a refusal: crashes are counted in
  `n_crashed` / `error_rate` and in schema validity, and excluded from the quality metrics.
- **schema validity**: share of final answers that round-trip the `AgentAnswer` contract,
  with crashes counted as invalid. **first-draft validity** is the share with no repair.
- **faithfulness**: the verifier's support ratio on the draft behind each answer. This is
  RAGAS-style faithfulness, except the claims come from the answer schema and entailment
  comes from the local NLI model instead of an LLM.
- **context precision / recall**: ID-based, on the chunks the tools returned. Precision is
  rank-weighted as in RAGAS (mean precision@k at the ranks of relevant chunks).
- **latency** p50/p95 (nearest rank, crashes excluded) and **cost** per question.

The judge (`evaluation/judge.py`, versioned prompts `evaluation/prompts/judge.md` and
`judge_refusal.md`) grades answers on correctness 1-3 against the reference answer,
faithfulness to the cited text and citation adequacy, and grades refusals on whether
refusing was appropriate. It sees exactly the text the human labeller sees, and each
verdict and label records a hash of that text, so a label is only compared with a verdict
on the same answer. On Ollama the judge gets the full `num_predict` budget (qwen3 thinks
before answering), and a truncated verdict is an error, not a grade.

**Judge calibration.** `eval/labels/sample.jsonl` is a frozen, committed sample of 60 items
built by `citeshield-label-sample` from stored runs: 36 real answers, 12 refusals (6
appropriate, 6 not) and 12 synthetic perturbations of real answers (a claim strengthened
beyond its source, a citation swapped to a non-expected provision, a fact contradicted).
The perturbations exist because the verifier removes most bad claims before an answer is
final, so real answers alone give almost no negative examples; they are labelled by a
person like every other item, and the labeller never sees which items are synthetic. The
sample is split by golden question (no question on both sides), 30 dev / 30 held-out.
Labelling instructions and worked examples: [eval/labels/GUIDE.md](eval/labels/GUIDE.md).
`citeshield-calibrate-judge` reports, per grade and split, Cohen's kappa
(quadratic-weighted for correctness), agreement, confusion matrices, human class counts and
a seeded bootstrap 95% interval. A grade "meets target" only if, on held-out, there are at
least 20 pairs with at least 5 human labels in each of two classes and both the kappa and
the lower end of its interval are >= 0.7; otherwise it is "below target" or "insufficient
data". Dev disagreements are listed for prompt iteration; held-out ones are only counted.
Results go to `eval/results/judge_calibration.md`. Labels come from one annotator.

**Model-quality gate** (`.github/workflows/eval-gate.yml`, started manually; not part of the
required PR checks). It runs the smoke subset against Claude and compares the result with
`eval/baselines/anthropic.json`. It fails if
faithfulness drops by more than 0.03 or falls below 0.85, citation precision drops by more
than 0.03, refusal F1 drops by more than 0.05, or schema validity is below 100%. A run made
on a different subset, golden set, corpus or set of models is reported as not comparable;
it never counts as a pass. Comparability also covers the Ollama model digest, the LLM
decoding options (`num_ctx`, `num_predict`) and the devices and weight dtypes the models
actually loaded with. Prompts and configs may differ from the baseline: they are what the
gate tests. The gate exits 0 (pass), 1 (regression) or 2 (not comparable / no baseline);
both 1 and 2 fail the job. The job also fails, with an explanation, if the
`ANTHROPIC_API_KEY` secret or the baseline file is missing, so it can never pass as a
no-op. Neither exists yet, so no model-quality gate has run.

**Gates per provider.** The thresholds above apply to Claude only. For local Ollama
(qwen3:4b) only the deterministic checks block: schema validity below 100% and any crash.
Faithfulness, citation precision and refusal F1 are reported as advisory (drops of more
than 0.10, 0.10 and 0.15 are flagged) and never fail a run, and the 0.85 faithfulness floor
does not apply. qwen3:4b samples, so the same question can be refused in one run and
answered in the next, and the local subset is small; the advisory thresholds stay
uncalibrated until run-to-run variance has been measured on repeated runs.

**Local Ollama baseline (provisional).** On the 16 GB dev machine the 40-question smoke run
completes but exhausts system RAM (diagnostic report:
[eval/results/ollama_smoke40_mixed_diagnostic.md](eval/results/ollama_smoke40_mixed_diagnostic.md);
faithfulness 0.841 there is evidence, not a threshold). The local baseline therefore uses
`--subset local`: 14 hand-picked questions in `eval/golden/smoke_local.json` (lookup,
multi-paragraph, cross-regulation, temporal, checklist, and three refusal types; the file
records the selection rule). It is stored as `eval/baselines/ollama_provisional.json`, with
its limitations in `baseline_info`. It is not statistically representative. The run
configuration keeps the GPU for the LLM and saves RAM:

```bash
CITESHIELD_LLM__PROVIDER=ollama CITESHIELD_LLM__OLLAMA_NUM_CTX=24576 \
CITESHIELD_LLM__OLLAMA_NUM_PREDICT=8192 CITESHIELD_RETRIEVER__EMBED_DTYPE=bfloat16 \
CITESHIELD_RETRIEVER__RERANK_BATCH_SIZE=8 CITESHIELD_VERIFIER__BATCH_SIZE=4 \
uv run citeshield-eval --subset local
uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json --baseline eval/baselines/ollama_provisional.json
```

The 40-question `--subset smoke` run stays available for manual or scheduled local runs on
a machine with more RAM; never mix its metrics with a `local` run (the gate reports them as
not comparable).

CI does not build the index. Build it locally and publish it as a release asset keyed by the
corpus hash (printed by `citeshield-index` and stored in `data/index/manifest.json`):

```bash
tar -czf index.tar.gz -C data index
gh release create index-<corpus_hash> index.tar.gz --title "Index <corpus_hash>" --notes "Qdrant index"
```

## Ingestion and the data contract

Sources are pinned in `data/corpus_manifest.yaml` and downloaded as Official Journal XHTML
from the Publications Office Cellar API (the EUR-Lex website blocks scripted access).

Each paragraph becomes one `ChunkRecord` (`src/citeshield/contracts/chunk.py`). A paragraph
that is too long is split into its points, (a), (b) and so on, and the lead-in sentence is
repeated on each one. Every record goes through three validation stages
(`src/citeshield/ingest/validate.py`):

1. **schema**: types, a deterministic `chunk_id`, a content hash, and official provenance.
2. **semantic**: the record agrees with the document's table of contents, and the paragraph
   text starts with its own number.
3. **collection**: no duplicate ids, and no gaps in paragraph numbering.

Records that fail go to `data/quarantine/<run_id>.jsonl` together with the reasons.
The run fails, and writes no corpus, if any document has structural errors or if
more than 2% of its records are quarantined. `data/processed/ingest_report.json` gives the
counts and the top quarantine reasons for each regulation.

## Retrieval

- **Encoder**: `BAAI/bge-m3`, implemented directly on `transformers` (`retrieval/embed.py`).
  One forward pass gives a dense vector (CLS) and a sparse lexical vector (from the model's
  `sparse_linear` head).
- **Index**: Qdrant stores two named vectors per chunk. The payload holds only ids and
  filterable metadata. Text always comes from the validated corpus, and the retriever
  refuses to run if the index manifest's corpus hash doesn't match `chunks.jsonl`.
- **Search**: dense and sparse prefetch, fused with RRF. Then `BAAI/bge-reranker-v2-m3`
  reranks the top 40 down to 8. Definitions and "referred to in paragraph N"
  cross-references are added as labelled expansions.
- **Recitals as context**: only binding text (articles and annexes) is ranked. The best 2
  recitals are attached separately as non-binding interpretive context.

### Results

The golden set (`eval/golden/questions.jsonl`) has 53 answerable questions with
paragraph-level expected citations and 7 questions the agent should refuse. The table is
generated by `citeshield-eval-retrieval`:

| variant | hit@1 | recall@3 | recall@5 | recall@8 | recall@20 | MRR@20 |
|---|---|---|---|---|---|---|
| dense | 0.585 | 0.796 | 0.852 | 0.909 | 0.984 | 0.740 |
| sparse | 0.472 | 0.676 | 0.871 | 0.918 | 0.975 | 0.665 |
| hybrid (RRF) | 0.585 | 0.786 | 0.871 | 0.918 | 0.994 | 0.742 |
| hybrid + rerank | **0.377** | 0.758 | 0.843 | 0.928 | 0.984 | 0.624 |
| hybrid, recitals as context | 0.792 | 0.890 | 0.956 | 0.975 | 0.994 | 0.870 |
| **hybrid + rerank, recitals as context** (default) | **0.811** | 0.890 | 0.947 | 0.975 | 0.994 | **0.886** |

What this shows:

- **Reranking alone made precision worse.** The cross-encoder prefers recitals, whose
  explanatory prose reads like an answer, over the binding article the user needs to cite.
  Ranking only operative text fixes this, and only then does the reranker help.
- **The reranker score is a usable refusal signal.** With the default config, every
  answerable question has a top score of at least 0.67, and every out-of-corpus question
  scores at most 0.15. The closest call is the Cyber Resilience Act question, which lands on
  NIS2 text. Milestone 5 calibrates the refusal threshold τ on this gap.
- **Remaining misses are cross-regulation questions**, such as GDPR vs NIS2 breach
  deadlines, where one regulation crowds out the other. The agent in milestone 4 is meant to
  fix this by retrieving per regulation.
- Definition and cross-reference expansion recovered no extra golden citations on this set.
  It stays enabled because it supplies context the generator needs, and its value will be
  measured on answer faithfulness rather than on recall.

Caveats: the set is small, and it was written by the system's author. The recital policy was
chosen after seeing these misses. It's a principled choice (recitals aren't binding), but it
needs confirming on the larger held-out golden set planned for milestone 6.
