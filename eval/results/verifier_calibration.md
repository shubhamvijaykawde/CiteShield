# Verifier NLI calibration

Model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. Pairs: `eval/verification/claims.jsonl`, labelled by the project author (not independent annotators); split stratified by label (alternate pairs).

Counts: `{"calibration": {"supported": 16, "unsupported": 8, "contradicted": 6}, "holdout": {"supported": 16, "unsupported": 8, "contradicted": 6}}`

**Entailment threshold 0.514**: midpoint of the calibration margin; any value in the open interval (0.038, 0.991) separates the calibration pairs equally well.

| split | accepted | false accepts | precision | recall |
|---|---|---|---|---|
| calibration | 16 | 0 | 1.000 | 1.000 |
| holdout | 16 | 0 | 1.000 | 1.000 |

**Contradiction threshold 0.563**: midpoint of the calibration margin; feasible interval (0.166, 0.959).

| split | flagged | supported flagged | precision | recall |
|---|---|---|---|---|
| calibration | 6 | 0 | 1.000 | 1.000 |
| holdout | 6 | 0 | 1.000 | 1.000 |

Held-out disagreements:

none

What this does and does not show: the model separates these pairs with a wide margin, so the data shows the classifier is reliable on clear-cut claims, but it does not determine the threshold precisely. The pairs are clear paraphrases or clear errors, written by the person who built the system; real drafts include borderline paraphrases (e.g. dropping a qualifier such as 'where feasible'), which this set does not contain. 60 pairs, single-chunk premises. Provisional until a harder, independently labelled set exists (milestone 6).
