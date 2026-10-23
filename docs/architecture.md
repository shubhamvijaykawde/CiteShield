# Architecture

How a question becomes a cited, verified answer or a typed refusal. Results and their
limits are in [evaluation.md](evaluation.md) and [RELEASE.md](../RELEASE.md).

```
browser UI ─► FastAPI (SSE) ─► guard ─► agent ⇄ tools ─► draft ─► validate ─► verify ─► policy ─► answer
                                (scope,   (retrieve,       (JSON     (schema,    (citations,  (drop/qualify/
                                 advice,   get_article,     schema)   ids)        quotes, NLI)  refuse)
                                 manipulation) compare, checklist)
```

| part | where | what |
|---|---|---|
| Ingestion | `src/citeshield/ingest/` | Official Journal XHTML, one chunk per paragraph, three validation stages, quarantine |
| Retrieval | `src/citeshield/retrieval/` | bge-m3 dense + sparse in embedded Qdrant, RRF fusion, bge-reranker-v2-m3 |
| Agent | `src/citeshield/agent/` | LangGraph state machine, four allowlisted read-only tools, structured `AgentAnswer` |
| Verifier | `src/citeshield/verify/` | citation, verbatim-quote and NLI entailment checks per claim and clause |
| Policy | `src/citeshield/agent/policy.py` | removes failed claims or refuses with a typed reason |
| Web app | `src/citeshield/api/` | FastAPI: static UI at `/`, `POST /ask` (SSE), `/health`, `/ready`, `/corpus`, `/sources/{id}` |
| Evaluation | `src/citeshield/evaluation/` | golden set, deterministic metrics, over-refusal and adversarial scoring, judge |
| Observability | `src/citeshield/observability.py` | optional Langfuse traces and MLflow evaluation runs |

## LLM providers

The graph, tools, answer contract and verifier are the same for both providers. Nothing
switches provider implicitly (`src/citeshield/agent/config.py`).

| | local Ollama (the demo) | Anthropic |
|---|---|---|
| selected by | `CITESHIELD_LLM__PROVIDER=ollama`; `scripts/start-demo.ps1` always sets it | the code default (`provider = "anthropic"`) when nothing overrides it |
| guard, research, draft | `qwen3:4b` for every call (`ollama_guard_model` defaults to the same model) | guard `claude-haiku-5-5`, agent and draft `claude-sonnet-5-5` |
| cost | none; no API key, the Anthropic client is never constructed | paid API calls; needs `ANTHROPIC_API_KEY` |
| structured draft | Ollama `format` with the `AgentAnswer` JSON schema (grammar-constrained) | `output_config.format` JSON schema |
| not available | prompt caching, effort levels, `tool_choice`, server-side refusal fallback | — |
| evaluated | yes: the 14-question local subset (single runs) | **no**: no Anthropic run or baseline exists |

Ollama settings that matter for qwen3:4b (`src/citeshield/agent/ollama.py`): thinking
cannot be turned off and counts against `num_predict` (8192), and `num_ctx` is always sent
because Ollama's 4096-token default silently truncates prompts; a prompt that would not fit
is refused before it is sent. The launcher uses `num_ctx` 24576.

## Agent

`src/citeshield/agent/graph.py` defines the agent as a LangGraph state machine:

```
guard ──in scope──► agent ⇄ tools ──► draft ──► validate ──► final answer
  │                 (≤ 6 tool rounds)              │  ▲
  └──out of scope──► typed refusal                 └──┘ one repair, then drop unsupported claims
```

- **guard** (structured output): screens out questions about other laws, requests to assess
  the user's own situation, and attempts to change the assistant's instructions. These are
  refused before any retrieval runs. If the guard fails, the question goes through to
  research; the guard saves cost and is not the security boundary.
- **agent ⇄ tools**: four read-only tools, `retrieve`, `get_article`, `compare_articles` and
  `generate_checklist`.
  - Each tool has a strict schema, and the arguments are validated again with Pydantic.
  - Calls to any other tool name are rejected and flagged.
  - Every chunk a tool returns goes into an evidence registry.
  - Source text is wrapped in `<source>` delimiters that the text itself can't close.
- **draft**: a separate call with a JSON-schema output writes the `AgentAnswer`. Sonnet 5.5
  can't be forced to call a tool, so the draft is a separate call whose JSON schema
  guarantees the format; on Ollama the same schema constrains the grammar.
- **validate**: checks the Pydantic contract, and checks that every citation is a chunk some
  tool returned in this run. On errors it sends the draft back once with the error list. If
  the repair also fails, claims with unresolved citations are dropped and the answer becomes
  `partial`, or a refusal if nothing is left. A draft that stays malformed falls back to a
  salvage refusal, which the evaluators count as invalid output, not as a refusal.
- **Modes.** Answer, compare and checklist all return the same `AgentAnswer`; compare and
  checklist are prompt hints over the same tools.
- **History is append-only.** Assistant turns go back to the API unchanged, including their
  thinking blocks.
- **Anthropic only: prompt caching and refusal fallback.** The system prompt and tool list
  are cached, and Sonnet calls use the server-side refusal fallback (`fallbacks: "default"`).
- **Logged with each run**: prompt versions (content hashes), the agent, retriever and corpus
  hashes, token usage and, for Anthropic, estimated cost.

## Verification and refusal policy

Every drafted answer is verified before anyone sees it (`src/citeshield/verify/`):

1. **Citations** are checked against the canonical corpus and this run's evidence:
   `fabricated` (no such id), `version_mismatch` (same provision, other `@version`),
   `non_citable` (e.g. an untrusted source), `not_retrieved` (no tool returned it).
2. **Quotes** must match the canonical text after deterministic normalisation (NFKC,
   typographic quotes/dashes, whitespace, case). Near misses are reported, never accepted.
3. **Entailment**: a matching quote only proves the passage exists. Each claim must be
   entailed by its cited text according to a local NLI model
   (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, pinned revision, CPU by
   default). Verdicts: supported, unsupported, contradicted, conflicting, citation_invalid.
   A claim that joins obligations ("must send X and notify Y", or "A; B") must also have
   each clause entailed. Whole-sentence entailment alone once let a claim through that
   was only half supported (`VerifierConfig.check_conjuncts`). Other forms of partial
   support, such as an unsupported qualifier inside one clause, are not split out.
4. **Summary and caveats** are split into sentences; only sentences entailed by the verified
   claims or their sources are kept. Disclaimers are added by code.

Failures get one revision with per-claim reasons. After that `agent/policy.py` removes what
failed (answer becomes `partial`) or refuses with a specific reason (`invalid_citations`,
`unsupported_claims`, `conflicting_sources`, or `insufficient_evidence` when the best
retrieval score is below 0.30). Verification can only lower a status. The verifier is a
required constructor argument: no agent can be built without it. The full
`VerificationReport` is returned in the final SSE event.

Thresholds (`src/citeshield/verify/config.py`):

- **Entailment 0.51, contradiction 0.56, summary sentences 0.51.** These come from
  `citeshield-calibrate-verifier` on 60 author-labelled claim/source pairs
  (`eval/verification/claims.jsonl`, stratified calibration/held-out split); see
  [verifier_calibration.md](../eval/results/verifier_calibration.md). The model separates
  that set with a wide margin, so the data shows reliability on clear-cut claims but does
  not pin the thresholds down. They are provisional.
- **Retrieval-score refusal 0.30.** On the 60-question retrieval benchmark (q001–q060),
  every answerable question's top rerank score was at least 0.668 and every unanswerable
  one at most 0.151. 0.30 sits in that gap, nearer the unanswerable side. It is provisional:
  the sample is small, and the agent's own queries differ from the golden questions.

Design rules:

- **The model never sets a verdict.** Citation validity, quote matching and entailment are
  computed from the corpus and an NLI model.
- **Sources are data, not instructions.** They sit in delimiters the text cannot close, and
  untrusted sources are never citable. These are prompt-level and pipeline defenses; they do
  not make the system injection-proof (see the adversarial results in
  [RELEASE.md](../RELEASE.md#adversarial-results-preliminary)).

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

The validated corpus (`data/processed/chunks.jsonl`) is committed, so the demo needs no
download from EUR-Lex. `citeshield-ingest` rebuilds it and only matters if the corpus
changes.

## Retrieval

- **Encoder**: `BAAI/bge-m3`, implemented directly on `transformers` (`retrieval/embed.py`).
  One forward pass gives a dense vector (CLS) and a sparse lexical vector (from the model's
  `sparse_linear` head).
- **Index**: Qdrant stores two named vectors per chunk. The payload holds only ids and
  filterable metadata. Text always comes from the validated corpus, and the retriever
  refuses to run if the index manifest's corpus hash doesn't match `chunks.jsonl`.
- **Search**: dense and sparse prefetch (40 each), fused with RRF. Then
  `BAAI/bge-reranker-v2-m3` reranks the top 40 down to 8. Definitions and "referred to in
  paragraph N" cross-references are added as labelled expansions.
- **Recitals as context**: only binding text (articles and annexes) is ranked. The best 2
  recitals are attached separately as non-binding interpretive context. The benchmark that
  motivated this choice is in [evaluation.md](evaluation.md#retrieval-benchmark).
