# CiteShield

A cited, injection-resistant agent that answers compliance questions over public EU legal
texts (GDPR, AI Act, DSA, NIS2, Data Act) and refuses when the evidence is thin.

> Not legal advice. CiteShield points you to the paragraphs that matter; it does not replace a lawyer.

## Status

| Milestone | State |
|---|---|
| 1. Scaffold, config, data contracts, CI | done |
| 2. Ingestion with data contract and quarantine | done |
| 3. Hybrid retrieval + reranker | done |
| 4. LangGraph agent + FastAPI streaming | built and tested offline; live run pending |
| 5. Claim-level verifier + refusal policy | next |
| 6–9. Evals, red-team, observability, delivery | planned |

## Quickstart

```bash
uv sync --group dev --extra ml      # CUDA torch on Windows, CPU torch on Linux
uv run citeshield-ingest            # fetch, parse, validate -> data/processed/chunks.jsonl
uv run citeshield-index             # embed with bge-m3 into embedded Qdrant (data/index)
uv run citeshield-search "Can I use emotion recognition on employees?"
uv run citeshield-eval-retrieval    # golden-set retrieval metrics -> eval/results/
uv run pytest

cp .env.example .env                # then set ANTHROPIC_API_KEY
uv run citeshield-ask "Are AI systems used to screen job applicants high-risk?"
uv run uvicorn citeshield.api.main:app   # POST /ask streams Server-Sent Events
```

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
