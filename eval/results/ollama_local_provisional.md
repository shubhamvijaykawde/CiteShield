# Answer eval `ollama-local-mixed-1`

14 items (local), provider `ollama`, agent `qwen3:4b`, corpus `b5afd60b188d1cfa`, golden `9aa8049d79c5`, 2026-10-22T19:32:59Z.

| metric | value |
|---|---|
| n_completed | 14 |
| n_crashed | 0 |
| citation_precision | 1.000 |
| citation_recall | 0.500 |
| refusal_precision | 0.375 |
| refusal_recall | 1.000 |
| refusal_f1 | 0.545 |
| schema_validity | 1.000 |
| first_draft_validity | 0.643 |
| faithfulness | 0.821 |
| context_precision | 0.697 |
| context_recall | 0.909 |
| error_rate | 0.000 |
| latency_p50_s | 187.6 |
| latency_p95_s | 454.1 |
| cost_per_question_usd | 0.0000 |
| judge_n | 0.000 |
| judge_correctness | n/a |
| judge_faithful | n/a |
| judge_citation_adequate | n/a |

Shares below are over each category's completed (non-crashed) items.

| category | n | crashed | refused | citation P | citation R | faithfulness |
|---|---|---|---|---|---|---|
| checklist | 2 | 0 | 0.00 | 1.000 | 1.000 | 0.875 |
| cross_regulation | 2 | 0 | 0.50 | 1.000 | 0.500 | 1.000 |
| lookup | 3 | 0 | 0.67 | 1.000 | 0.333 | 1.000 |
| multi_paragraph | 2 | 0 | 0.50 | 1.000 | 0.250 | 1.000 |
| temporal | 2 | 0 | 0.50 | 1.000 | 0.500 | 0.500 |
| unanswerable | 3 | 0 | 1.00 | n/a | n/a | n/a |

Models: `{"llm_provider": "ollama", "agent_model": "qwen3:4b", "guard_model": "qwen3:4b", "embed_model": "BAAI/bge-m3", "rerank_model": "BAAI/bge-reranker-v2-m3", "nli_model": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "nli_revision": "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7", "ollama_digests": {"qwen3:4b": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"}}`.

LLM options: `{"num_ctx": 24576, "num_predict": 8192, "think": true}`; num_ctx observed [24576], consistent: True.

Config hashes: `{"agent": "77adc77a4b0b", "llm": "d8ddfd35576b", "retriever": "17f538090a82", "verifier": "b2f7e71fc57f"}`.

Code: `{"git_commit": "23456aea02da79eb1e87a0e481c854511217fb32", "git_dirty": true, "ollama_version": "0.40.7"}`.
