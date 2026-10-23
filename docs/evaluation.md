# Evaluation

How CiteShield is measured, on which question sets, and how far each result can be
trusted. The verified numbers for the 0.1.0 release are in [RELEASE.md](../RELEASE.md);
detailed reports are in [`eval/results/`](../eval/results/).

## Question sets

Results come from different populations. Never compare numbers across rows.

| set | file | size | used for |
|---|---|---|---|
| Golden set | `eval/golden/questions.jsonl` | 151 questions: 121 answerable, 30 to refuse | the answer evaluation (`--subset full`) and the source of every subset below |
| Retrieval benchmark | q001–q060 of the golden set, as it was in Milestone 3 | 60 questions: 53 answerable, 7 to refuse | the retrieval table below and the 0.30 retrieval-score refusal threshold; run before the set grew to 151 |
| Smoke subset | `eval/golden/smoke.json` | 40 questions: 32 answerable, 8 to refuse | the Claude model-quality gate; too heavy for the 16 GB dev laptop on Ollama |
| Local subset | `eval/golden/smoke_local.json` | 14 questions: 11 answerable, 3 to refuse | every reported local qwen3:4b answer and over-refusal result |
| Adversarial cases | `eval/adversarial/cases.json` | 5 poisoned-content cases | the preliminary adversarial results |
| Verifier calibration | `eval/verification/claims.jsonl` | 60 author-labelled claim/source pairs | the NLI thresholds |
| Judge calibration | `eval/labels/sample.jsonl` | 60 items from stored runs, 30 dev / 30 held-out | the LLM-judge calibration (failed) |

The golden questions cover lookups, definitions, multi-paragraph, cross-regulation,
comparison, checklist and temporal questions, plus 30 (20%) that must be refused (other
laws, case law, enforcement facts, personal legal advice, injection, off-topic). Each
answerable item lists its expected citations at paragraph level and a short reference
answer. Items q061–q151 were drafted with every citation checked against the corpus and are
marked `"reviewed": false` until a human has read them. The reference answers of q001–q053
were also drafted from the text of their expected provisions and still need a human read.

The local subset's selection rule is recorded in the file: fixed before running, without
looking at any item's past results, preferring the human-reviewed items q001–q060. It is
not statistically representative.

## Commands

Run from PowerShell in the repository folder. `py -m uv` works without `uv` on `PATH`.

```powershell
py -m uv run citeshield-eval --subset local                        # 14 questions, calls the LLM
py -m uv run citeshield-eval --subset smoke                        # 40 questions, calls the LLM
py -m uv run citeshield-eval --subset full --judge                 # every item, plus the LLM judge
py -m uv run citeshield-eval --rescore eval/runs/<run_id>          # re-score, no agent calls
py -m uv run citeshield-eval-refusal eval/runs/<run_id>            # over-refusal, offline
py -m uv run citeshield-eval-adversarial --score eval/runs/<run_id>   # attack scoring, offline
py -m uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json  # compare with the baseline
py -m uv run citeshield-eval-retrieval                             # retrieval benchmark (needs the index)
py -m uv run citeshield-label                                      # hand-label the frozen sample
py -m uv run citeshield-label --coverage                           # labelled vs. remaining
py -m uv run citeshield-calibrate-judge                            # judge vs. labels, kappa + report
py -m uv run citeshield-calibrate-verifier                         # NLI threshold calibration
```

Without overrides, `citeshield-eval` uses the configured provider, which defaults to
Anthropic and makes paid API calls. For local runs, set the Ollama provider first. This is
the configuration of the reported local runs; it keeps the GPU for the LLM and saves RAM:

```powershell
$env:CITESHIELD_LLM__PROVIDER = "ollama"; $env:CITESHIELD_LLM__OLLAMA_NUM_CTX = "24576"
$env:CITESHIELD_LLM__OLLAMA_NUM_PREDICT = "8192"; $env:CITESHIELD_RETRIEVER__EMBED_DTYPE = "bfloat16"
$env:CITESHIELD_RETRIEVER__RERANK_BATCH_SIZE = "8"; $env:CITESHIELD_VERIFIER__BATCH_SIZE = "4"
py -m uv run citeshield-eval --subset local
py -m uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json --baseline eval/baselines/ollama_provisional.json
```

Add `--mlflow` to `citeshield-eval`, `citeshield-eval-refusal` or
`citeshield-eval-adversarial --score` to log the run (see
[development.md](development.md#observability-optional)).

A run writes `eval/runs/<run_id>/` (`meta.json`, `answers.jsonl`, `judge.jsonl`,
`metrics.json`, `report.md`; gitignored) and copies the report to
[eval/results/answers_latest.md](../eval/results/answers_latest.md). Re-running the same
`--run-id` resumes where it stopped, and refuses if the agent configuration changed in
between. `--write-smoke` regenerates the smoke subset deterministically (proportional per
category, at least two items each); changing it requires a new baseline.

## Metrics

All deterministic (`src/citeshield/evaluation/metrics.py` and `refusal_eval.py`). The
answer-quality metrics are computed per question and averaged over the questions where
they are defined; crashed questions are excluded.

- **Citation precision / recall**: the share of a question's cited chunks that match an
  expected paragraph (answerable questions with at least one citation), and the share of
  its expected paragraphs that were cited (answerable questions).
- **Refusal precision / recall / F1**: a deliberate refusal is the positive class. A crash
  (no answer at all, e.g. a context overflow) is never a refusal: crashes are counted in
  `n_crashed` / `error_rate` and in schema validity, and excluded from the quality metrics.
- **Over-refusal**: answerable questions refused, out of answerable questions.
- **Coverage**: answered or partial, out of answerable questions.
- **Invalid output**: a draft that never became a valid answer, including a salvage
  fallback after a failed repair. Never counted as a refusal.
- **Correct refusal / under-refusal**: out of the questions that should be refused.
- **Schema validity**: share of final answers that round-trip the `AgentAnswer` contract,
  with crashes counted as invalid. **First-draft validity** is the share with no repair.
- **Faithfulness**: the verifier's support ratio on the draft behind each answer, averaged
  over the questions that have a verified draft. This is RAGAS-style faithfulness, except
  the claims come from the answer schema and entailment comes from the local NLI model
  instead of an LLM.
- **Context precision / recall**: ID-based, on the chunks the tools returned, per
  answerable question. Precision is rank-weighted as in RAGAS (mean precision@k at the
  ranks of relevant chunks).
- **Latency** p50/p95 (nearest rank, crashes excluded) and **cost** per question (Anthropic
  only; Ollama does not bill).

## Retrieval benchmark

Produced by `citeshield-eval-retrieval` in Milestone 3 on the 60-question retrieval
benchmark (q001–q060: 53 answerable questions with paragraph-level expected citations, 7
to refuse). It has not been re-run on the 151-question golden set. Source:
[retrieval_latest.md](../eval/results/retrieval_latest.md). Ranking metrics use the 53
answerable questions:

| variant | hit@1 | recall@3 | recall@5 | recall@8 | recall@20 | MRR@20 |
|---|---|---|---|---|---|---|
| dense | 0.585 | 0.796 | 0.852 | 0.909 | 0.984 | 0.740 |
| sparse | 0.472 | 0.676 | 0.871 | 0.918 | 0.975 | 0.665 |
| hybrid (RRF) | 0.585 | 0.786 | 0.871 | 0.918 | 0.994 | 0.742 |
| hybrid + rerank | **0.377** | 0.758 | 0.843 | 0.928 | 0.984 | 0.624 |
| hybrid, recitals as context | 0.792 | 0.890 | 0.956 | 0.975 | 0.994 | 0.870 |
| **hybrid + rerank, recitals as context** (default) | **0.811** | 0.890 | 0.947 | 0.975 | 0.994 | **0.886** |

- **hit@1**: share of the 53 questions whose top result matches an expected provision.
- **recall@k**: mean over the 53 questions of the share of each question's expected
  provisions among the top k.
- **MRR@20**: mean of 1/rank of the first relevant result, 0 if none in the top 20.

What this shows:

- **Reranking alone made precision worse.** The cross-encoder prefers recitals, whose
  explanatory prose reads like an answer, over the binding article the user needs to cite.
  Ranking only operative text fixes this, and only then does the reranker help.
- **The reranker score is a usable refusal signal.** With the default config, every
  answerable question has a top score of at least 0.67, and every out-of-corpus question
  scores at most 0.15. The closest call is the Cyber Resilience Act question, which lands on
  NIS2 text. The verifier's 0.30 `insufficient_evidence` threshold was set in this gap.
- **Remaining misses are cross-regulation questions**, such as GDPR vs NIS2 breach
  deadlines, where one regulation crowds out the other. The agent's system prompt asks for
  one `retrieve` call per regulation in such questions; that mitigation has not been
  measured as a retrieval metric.
- Definition and cross-reference expansion recovered no extra golden citations on this set.
  It stays enabled because it supplies context the generator needs.

Caveats: the set is small, and it was written by the system's author. The recital policy was
chosen after seeing these misses. It's a principled choice (recitals aren't binding), but it
has not been confirmed on the larger golden set.

## LLM judge

The judge (`evaluation/judge.py`, versioned prompts `evaluation/prompts/judge.md` and
`judge_refusal.md`) grades answers on correctness 1-3 against the reference answer,
faithfulness to the cited text and citation adequacy, and grades refusals on whether
refusing was appropriate. It sees exactly the text the human labeller sees, and each
verdict and label records a hash of that text, so a label is only compared with a verdict
on the same answer. On Ollama the judge gets the full `num_predict` budget (qwen3 thinks
before answering), and a truncated verdict is an error, not a grade.

**Calibration.** `eval/labels/sample.jsonl` is a frozen, committed sample of 60 items
built by `citeshield-label-sample` from stored runs: 36 real answers, 12 refusals (6
appropriate, 6 not) and 12 synthetic perturbations of real answers (a claim strengthened
beyond its source, a citation swapped to a non-expected provision, a fact contradicted).
The perturbations exist because the verifier removes most bad claims before an answer is
final, so real answers alone give almost no negative examples; they are labelled by a
person like every other item, and the labeller never sees which items are synthetic. The
sample is split by golden question (no question on both sides), 30 dev / 30 held-out.
Labelling instructions and worked examples: [eval/labels/GUIDE.md](../eval/labels/GUIDE.md).

`citeshield-calibrate-judge` reports, per grade and split, Cohen's kappa
(quadratic-weighted for correctness), agreement, confusion matrices, human class counts and
a seeded bootstrap 95% interval. A grade "meets target" only if, on held-out, there are at
least 20 pairs with at least 5 human labels in each of two classes and both the kappa and
the lower end of its interval are >= 0.7; otherwise it is "below target" or "insufficient
data". Dev disagreements are listed for prompt iteration; held-out ones are only counted.
Labels come from one annotator.

**Result: the local qwen3:4b judge failed calibration** on every answer grade (held-out
kappa 0.694 correctness, 0.391 faithful, 0.500 citation adequate; see
[judge_calibration.md](../eval/results/judge_calibration.md)). It is informational only and
never gates anything. Its held-out half has been used for this prompt version, so a new
judge needs freshly labelled held-out items.

## Model-quality gate

`.github/workflows/eval-gate.yml` is started manually and is not part of the required PR
checks. It runs the smoke subset against Claude and compares the result with
`eval/baselines/anthropic.json`. It fails if faithfulness drops by more than 0.03 or falls
below 0.85, citation precision drops by more than 0.03, refusal F1 drops by more than 0.05,
or schema validity is below 100%.

A run made on a different subset, golden set, corpus or set of models is reported as not
comparable; it never counts as a pass. Comparability also covers the Ollama model digest,
the LLM decoding options (`num_ctx`, `num_predict`) and the devices and weight dtypes the
models actually loaded with. Prompts and configs may differ from the baseline: they are
what the gate tests. The gate exits 0 (pass), 1 (regression) or 2 (not comparable / no
baseline); both 1 and 2 fail the job. The job also fails, with an explanation, if the
`ANTHROPIC_API_KEY` secret or the baseline file is missing, so it can never pass as a
no-op. **Neither exists yet, so no model-quality gate has run.**

**Gates per provider.** The thresholds above apply to Claude only. For local Ollama
(qwen3:4b) only the deterministic checks block: schema validity below 100% and any crash.
Faithfulness, citation precision and refusal F1 are reported as advisory (drops of more
than 0.10, 0.10 and 0.15 are flagged) and never fail a run, and the 0.85 faithfulness floor
does not apply. qwen3:4b samples, so the same question can be refused in one run and
answered in the next, and the local subset is small; the advisory thresholds stay
uncalibrated until run-to-run variance has been measured on repeated runs.

## Local Ollama baseline (provisional)

On the 16 GB dev machine the 40-question smoke run completes but exhausts system RAM
(diagnostic report:
[ollama_smoke40_mixed_diagnostic.md](../eval/results/ollama_smoke40_mixed_diagnostic.md);
its faithfulness of 0.841 is evidence, not a threshold). The local baseline therefore uses
`--subset local`, stored as `eval/baselines/ollama_provisional.json` with its limitations in
`baseline_info`. It predates the coordinated-clause check, so the gate reports new runs as
not comparable until it is re-recorded.

The 40-question `--subset smoke` run stays available for a machine with more RAM; never mix
its metrics with a `local` run (the gate reports them as not comparable).

## Publishing the index for CI

CI does not build the index. Build it locally and publish it as a release asset keyed by the
corpus hash (printed by `citeshield-index` and stored in `data/index/manifest.json`), using
the GitHub CLI:

```powershell
tar -czf index.tar.gz -C data index
gh release create index-<corpus_hash> index.tar.gz --title "Index <corpus_hash>" --notes "Qdrant index"
```

`tar` ships with Windows 10 and later.
