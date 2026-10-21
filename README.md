# CiteShield

A cited, injection-resistant agent that answers compliance questions over public EU legal
texts (GDPR, AI Act, DSA, NIS2, Data Act) and refuses when the evidence is thin.

> Not legal advice. CiteShield points you to the paragraphs that matter; it does not replace a lawyer.

## Status

| Milestone | State |
|---|---|
| 1. Scaffold, config, data contracts, CI | done |
| 2. Ingestion with data contract and quarantine | done |
| 3. Hybrid retrieval + reranker | next |
| 4–9. Agent, verifier, evals, red-team, observability, delivery | planned |

## Quickstart

```bash
uv sync --group dev
uv run citeshield-ingest            # fetch, parse, validate -> data/processed/chunks.jsonl
uv run pytest
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
