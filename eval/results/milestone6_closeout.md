# Milestone 6 closeout: answer evaluation, LLM judge, CI gate

Branch `milestone-6-eval`; checkpoint commit `16d8755` plus uncommitted labelling and
calibration work. Status: **closed with documented gaps**. The evaluation harness is in
place and measured; the local judge failed calibration and the Claude baseline does not
exist yet. Every number below comes from a file in this repository (named in each section),
except the resource measurements marked "session measurement", which come from the
resource logs of the runs made on 2026-10-22/23 and are not stored in the repository.

## 1. What was built

| Component | Where | What it does |
|---|---|---|
| Golden set | `eval/golden/questions.jsonl` | 151 questions (lookup, definition, multi-paragraph, cross-regulation, comparison, checklist, temporal; 30 must be refused), paragraph-level expected citations, reference answers. q061-q151 and the q001-q053 reference answers are marked or documented as not yet human-reviewed. |
| Subsets | `eval/golden/smoke.json`, `smoke_local.json` | 40-question stratified smoke subset (deterministic); 14-question local subset with a recorded selection rule. |
| Eval runner | `evaluation/run_eval.py`, `records.py` | Runs the agent per question, resumable; records the answer, cited sources and cost; run identity covers models (incl. Ollama digest), `num_ctx`/`num_predict`, model devices and loaded dtypes, config hashes and git state. |
| Metrics | `evaluation/metrics.py` | Citation precision/recall, refusal precision/recall/F1, schema validity, first-draft validity, faithfulness (verifier support ratio), context precision/recall, latency, cost. Crashes are counted separately (`n_crashed`) and never scored as refusals. |
| Gate | `evaluation/gate.py`, `.github/workflows/eval-gate.yml` | Provider-specific rules (section 6); refuses non-comparable runs; provisional baselines carry their limitations. |
| Judge | `evaluation/judge.py`, `prompts/judge.md`, `prompts/judge_refusal.md` | Grades answers (correctness 1-3, faithful, citation adequate) and refusals (appropriate). Frozen in `eval/labels/frozen_judge.json`. |
| Labelling | `evaluation/annotations.py`, `label_sample.py`, `labeling.py`, `eval/labels/GUIDE.md` | Frozen 60-item sample, blinded CLI, label validation, `--item` relabel, coverage. |
| Calibration | `evaluation/judge_calibration.py` | Kappa (quadratic for correctness), raw agreement, confusion matrices, seeded bootstrap 95% CIs, conservative verdicts, refuses an unfrozen judge. |

RAGAS itself is not used: faithfulness and context precision/recall are RAGAS-style metrics
computed from the answer schema, the local NLI verifier and chunk ids.

## 2. Labels and the judge-calibration sample

Source: `eval/labels/sample.jsonl`, `human_labels.jsonl`, `citeshield-label --coverage`.

- 60 items, split by golden question (no question on both sides): 30 dev, 30 held-out. Each
  half: 17 answered, 1 partial, 3 correct refusals, 3 wrong refusals, 6 synthetic
  perturbations (2 unsupported claim, 2 wrong citation, 2 contradiction).
- **Coverage 60/60** active labels, 60 unique item ids, 0 validation problems, one
  annotator (the project author). The label file is append-only: 61 lines, one historical
  duplicate (`q031-cdf5c90b1feb`, line 4: the original 3/no/yes, superseded by line 31:
  1/no/no after review; the answer reverses "shall contain" to "shall not contain").
- Held-out labels were entered by the author from the blinded packet
  `eval/labels/holdout_review_packet.md` (views byte-identical to the CLI).

## 3. Held-out agreement of the frozen judge

Source: `eval/results/judge_calibration.md` / `.json` (recomputed from cached verdicts and
active labels on 2026-10-23: identical). Judge: qwen3:4b (Ollama 0.40.7), answer prompt
`ac2a420f`, refusal prompt `46212112`. Target: kappa >= 0.7 and the lower end of its 95%
interval >= 0.7, with >= 20 pairs and >= 5 human labels in each of two classes.

| grade | held-out n | kappa | 95% CI | raw agreement | verdict |
|---|---|---|---|---|---|
| correctness | 24 | 0.694 quadratic (0.566 unweighted) | 0.43 to 0.89 | 0.750 | below target |
| faithful | 24 | 0.391 | -0.01 to 0.74 | 0.708 | below target |
| citation adequate | 24 | 0.500 | 0.14 to 0.82 | 0.792 | below target |
| refusal appropriate | 6 | 1.000 | 1.00 to 1.00 | 1.000 | insufficient data |

Held-out confusion (rows human, columns judge): correctness `[[1,2,0],[0,8,3],[0,1,9]]`,
faithful `[[6,4],[3,11]]`, citation adequate `[[4,5],[0,15]]`, refusal `[[3,0],[0,3]]`.
Error directions: correctness judge higher than human 5, lower 1; citation adequacy 5
lenient, 0 strict; faithfulness 4 lenient, 3 strict. 9 held-out items disagree on at least
one grade (counted, deliberately not listed).

Dev agreement for the same frozen judge (optimistic: the answer prompt was revised on dev,
`4a71b1d4` -> `ac2a420f`): correctness 0.541 (0.24-0.77), faithful 0.676 (0.38-0.92),
citation adequate 0.565 (0.18-0.90), refusal 6/6. Faithfulness falling from 0.676 on dev to
0.391 on held-out shows the dev revision did not generalise.

**Conclusions.**

- **The qwen3:4b judge is not calibrated for automatic quality gating.** None of the
  three answer grades meets the target; its judge scores must stay informational.
- **The held-out half has been consumed for prompt version `ac2a420f`/`46212112`.** Any
  further prompt change needs newly labelled held-out items for a fair measurement;
  re-scoring these 30 after a change would only measure fitting.
- Intervals are wide (24 pairs): the data rules out "well calibrated" but cannot pin the
  agreement down. Correctness grade 1 has 3 held-out labels (below the 5-label minimum)
  and refusals have 6 pairs per split.

## 4. Answer quality of the local agent (qwen3:4b)

Sources: `eval/results/ollama_local_provisional.md`, `ollama_smoke40_mixed_diagnostic.md`,
`ollama_smoke_failed_ctx16384.md`.

| run | n | crashed | citation P / R | refusal P / R / F1 | faithfulness | context recall | latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| local subset (provisional baseline) | 14 | 0 | 1.000 / 0.500 | 0.375 / 1.000 / 0.545 | 0.821 | 0.909 | 188 s / 454 s |
| 40-question smoke (diagnostic) | 40 | 0 | 0.944 / 0.484 | 0.364 / 1.000 / 0.533 | 0.841 | 0.938 | 178 s / 294 s |
| 40-question smoke at num_ctx 16384 (failed, diagnostic) | 40 | 17 | - | - | - | - | - |

The dominant product defect is **over-refusal**: in the 40-question run 22 questions were
refused, of which only 8 had to be (14 of 32 answerable questions refused), while
retrieval returned on average 94% of the expected provisions (context recall 0.938). Every
question that must be refused was refused (refusal recall 1.000), including the direct
prompt injection q146. Citations that are given are almost always right (precision
0.944-1.000); recall is about half.

## 5. Resource limitations (16 GB RAM, RTX 4060 Laptop 8 GB, Ryzen 7 7435HS)

- qwen3:4b at `num_ctx` 16384 ran out of context on 17/40 questions; 24576 fits on the GPU
  only when the embedder, reranker and NLI model run on CPU.
- With those on CPU the eval process reserves about 10 GB (session measurement), and system
  RAM ran nearly out: the provisional baseline's notes record 28 MiB available at the low
  point of the 14-question run; the 40-question run reached 1 MiB (session measurement).
  The embedder in bfloat16 (opt-in `embed_dtype`, see `retrieval/config.py`) saves about
  1.1 GB with identical rankings on the fixed benchmark; BF16 for the reranker and NLI
  model was 7-12x slower on this CPU, which has no native BF16 (session measurement).
- Judge runs load only the LLM and are not RAM-bound: VRAM free >= 2.3 GB and about 48 s
  per item, 1,453 s for the 30 held-out items (session measurements).
- Local latency is 3-8 minutes per question, so the 40-question smoke run takes about
  2.5 hours.

## 6. Provisional baseline and gate behaviour

- `eval/baselines/ollama_provisional.json` (from run `ollama-local-mixed-1`, `--subset
  local`): 14 questions, one run, labelled provisional with its limitations in
  `baseline_info`. Its metadata records commit `23456ae` with `git_dirty: true` as it was;
  the run's source files are those committed in `16d8755`. It is not statistically
  representative.
- Gate (`evaluation/gate.py`):
  - Claude: fails on a faithfulness drop > 0.03 or faithfulness < 0.85, a citation
    precision drop > 0.03, a refusal F1 drop > 0.05, or schema validity < 100%.
  - Ollama: only schema validity < 100% and any crash block; quality drops are advisory;
    the 0.85 floor does not apply.
  - A run is comparable only with the same subset and golden items, corpus, models (incl.
    Ollama digest), decoding options, devices and dtypes; otherwise it is "not comparable"
    and never passes. `--write-baseline` refuses runs without runtime metadata, with mixed
    options, or with crashes.
- `eval/baselines/anthropic.json` does not exist (no API key on the dev machine), so the
  `eval-gate` workflow skips with a notice. **CI has no blocking answer-quality gate yet.**

## 7. Remaining gaps

1. Claude baseline and a calibrated Claude judge (needs an API key); the CI gate is inert
   until then.
2. The local judge fails calibration; a new judge or prompt needs fresh held-out labels.
3. One annotator, no inter-annotator agreement; refusal and correctness-grade-1 coverage
   is thin.
4. q061-q151 and the q001-q053 reference answers still need a human review pass.
5. No repeated runs: run-to-run variance of qwen3:4b is unmeasured, so the Ollama advisory
   thresholds are uncalibrated.
6. The full 40-question local run is RAM-bound on the dev machine; it stays available for
   manual or scheduled runs on bigger hardware.

## 8. Recommended next milestone

**Milestone 7: adversarial and refusal-behaviour evaluation with deterministic checks**
(the planned red-team milestone, scoped to what this repository can measure today).

Why this one: it targets the product's core promises, evidence-backed answers, correct
refusals and injection resistance, and it does not depend on the uncalibrated judge.
Attacker-goal checks (canary strings, citations of poisoned chunks, non-allowlisted tool
calls, planted false obligations) and refusal outcomes are deterministic. The golden set
already supplies the benign-utility side.

Scope, in order:

1. **Over-refusal first**, because attack success measured on an agent that refuses 44%
   of answerable questions is meaningless. Trace why the 14 answerable smoke questions
   were refused despite context recall 0.94 (guard label, verifier support ratio,
   retrieval-score threshold `min_retrieval_score`, draft refusal), using the stored
   records; fix the cause. Success measure: refusal precision on the local and smoke
   subsets, with refusal recall kept at 1.0.
2. Poisoned-chunk corpus with an `untrusted` trust tier and AgentDojo-style cases (corpus,
   tool-output and query vectors), with deterministic attacker-goal checkers.
3. "No defenses" baseline, then defenses one at a time (provenance, spotlighting,
   injection detector, tool allowlist, output sanitiser, verifier backstop): attack success
   rate and benign utility per configuration, as an ablation table.
4. A small red-team smoke set in CI with all defenses on, gated deterministically.
