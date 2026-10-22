# Answer eval `ollama-smoke-baseline`

40 items (smoke), provider `ollama`, agent `qwen3:4b`, corpus `b5afd60b188d1cfa`, golden `392e3bfd501a`, 2026-10-22T11:16:58Z.

| metric | value |
|---|---|
| n_completed | 23 |
| n_crashed | 17 |
| citation_precision | 0.833 |
| citation_recall | 0.567 |
| refusal_precision | 0.615 |
| refusal_recall | 1.000 |
| refusal_f1 | 0.762 |
| schema_validity | 0.575 |
| first_draft_validity | 0.783 |
| faithfulness | 0.909 |
| context_precision | 0.827 |
| context_recall | 0.967 |
| error_rate | 0.425 |
| latency_p50_s | 98.1 |
| latency_p95_s | 312.2 |
| cost_per_question_usd | 0.0000 |
| judge_n | 0.000 |
| judge_correctness | n/a |
| judge_faithful | n/a |
| judge_citation_adequate | n/a |

Shares below are over each category's completed (non-crashed) items.

| category | n | crashed | refused | citation P | citation R | faithfulness |
|---|---|---|---|---|---|---|
| checklist | 2 | 1 | 0.00 | 1.000 | 1.000 | 1.000 |
| comparison | 2 | 0 | 0.50 | 0.333 | 0.250 | 1.000 |
| cross_regulation | 2 | 1 | 0.00 | 1.000 | 1.000 | 1.000 |
| definition | 2 | 0 | 1.00 | n/a | 0.000 | n/a |
| lookup | 18 | 12 | 0.17 | 0.800 | 0.667 | 0.833 |
| multi_paragraph | 4 | 3 | 1.00 | n/a | 0.000 | n/a |
| temporal | 2 | 0 | 0.00 | 1.000 | 1.000 | 1.000 |
| unanswerable | 8 | 0 | 1.00 | n/a | n/a | n/a |

Crashes (17), excluded from quality metrics:

- q010: LLMContextExceeded: prompt is ~8360 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q014: LLMContextExceeded: prompt is ~8198 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q020: LLMContextExceeded: prompt is ~9802 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q025: LLMContextExceeded: prompt is ~8570 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q027: LLMContextExceeded: prompt is ~8492 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q031: LLMContextExceeded: prompt is ~8885 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q037: LLMContextExceeded: prompt is ~8877 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q039: LLMContextExceeded: prompt is ~8205 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q043: LLMContextExceeded: prompt is ~8329 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q053: LLMContextExceeded: prompt is ~11512 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITES
- q071: LLMContextExceeded: prompt is ~8449 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q085: LLMContextExceeded: prompt is ~8720 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q094: LLMContextExceeded: prompt is ~9927 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q097: LLMContextExceeded: prompt is ~8547 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q103: LLMContextExceeded: prompt is ~8814 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q104: LLMContextExceeded: prompt is ~8729 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
- q122: LLMContextExceeded: prompt is ~9082 tokens but only 8192 of num_ctx=16384 are left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower CITESH
