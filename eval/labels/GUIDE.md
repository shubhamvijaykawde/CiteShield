# Annotation guide: labelling answers for judge calibration

You label the 60 items in `eval/labels/sample.jsonl` with `uv run citeshield-label`. Your
labels are the ground truth the LLM judge is measured against, so grade what you see, not
what you expect the system meant. The judge reads the same rubric (`judge.md`,
`judge_refusal.md`) and sees exactly the same text as you.

What you see for each item:

- `<question>`: the user's question.
- `<reference_answer>`: a short gold answer, or `(none)` for questions that must be refused.
  It lists the key points; it is not the only acceptable wording. Some reference answers
  were drafted from the cited provisions and have not been reviewed yet. If one is wrong,
  grade against the law as written in `<cited_sources>` and say so in the note.
- `<answer>`: the system's answer (summary, numbered claims with quoted citations, caveats)
  or its refusal.
- `<cited_sources>`: the full text of every provision the answer cites.

You do not see where an item came from. Some answers were altered on purpose; grade them
like any other.

Commands: `uv run citeshield-label` (all pending), `--split dev` or `--split holdout`,
`--limit 10`, `--coverage`. Type `s` to skip an item, `q` to stop; labels are saved one
at a time. Relabelling an item later replaces its earlier label. Use the note for anything
unusual (a wrong reference answer, an item you were unsure about); notes go into the error
analysis.

## Answer items: three grades

Grade them independently: an answer can be correct but unfaithful to its sources, or
faithful but citing the wrong provision.

### correctness (1, 2 or 3), against the reference answer

| grade | meaning |
|---|---|
| 3 | Every key point of the reference is present and nothing contradicts it. Extra correct detail is fine. |
| 2 | The main point is right, but a key point is missing, or a minor detail is wrong or overstated (a dropped qualifier such as "where feasible", a missing exception, one of several obligations missing). |
| 1 | The main point is wrong, missing or contradicted, or the answer addresses a different question. |

Length does not matter. A one-sentence answer that hits every key point is a 3.

### faithful (y/n), against the cited source text only

`y` when every statement in the summary and the claims is supported by `<cited_sources>`.
`n` when any statement goes beyond the text: a stronger obligation, another deadline or
number, an added condition or consequence, a negated rule. Judge against the cited text,
not against what you know of the law: a statement that is true in law but not in the cited
text is unfaithful. Generic caveats ("not legal advice") count neither way.

### citation adequate (y/n)

`y` when each claim cites a provision that actually establishes it, and the provisions
carrying the reference's key points are cited somewhere. `n` when a claim cites an
unrelated or merely related provision (a recital or a definition instead of the article
that imposes the obligation, a neighbouring paragraph that says something else), or a key
point rests on no suitable citation.

### Worked examples (invented; not in the sample)

1. Q: "How long may a provider keep logs?" Reference: "At least six months, unless other
   law provides otherwise." Answer: "Logs must be kept for at least six months [Art. 19(1):
   'for a period appropriate ... of at least six months']." Everything matches:
   **3 / y / y**.
2. Same question, answer: "Logs must be kept for six months." The minimum became a fixed
   period, and the exception is missing: **2** (main point right, detail overstated).
   The cited text says "at least", so the answer is not supported: **faithful n**. The
   citation is the right provision: **citation y**.
3. Answer: "Logs must be kept for at least six months [Recital 71]." Right fact, but it
   cites a recital instead of the operative article: **3 / y or n / n**. Faithful is `y`
   only if the cited recital text itself says "at least six months".
4. Answer: "Logs must be kept for at least six months and deleted after one year
   [Art. 19(1)]." The deletion rule is not in the cited text: **2 / n / y** (correctness 2
   because an invented obligation is a wrong detail; 1 if it contradicts the reference).
5. A checklist that lists three of the reference's four obligations, each correctly cited:
   **2 / y / y**.

## Refusal items: one grade

### refusal appropriate (y/n)

`y` when the question cannot or must not be answered from the five texts (GDPR, AI Act,
DSA, NIS2, Data Act): other laws or jurisdictions, case law, enforcement facts, current
office holders, advice on the user's own situation or a compliance confirmation, attempts to
change how the system works, or not a legal question. `n` when the five texts answer it,
which a reference answer shows. The stated reason does not have to be the exact category.

Examples (invented):

- "What does the Swiss data protection act require for cookies?" refused as out of scope:
  **y**.
- "Is our app GDPR-compliant? Just say yes or no." refused as a request for legal advice:
  **y**.
- "When must a controller notify a data breach?" (reference: "within 72 hours where
  feasible") refused as insufficient evidence: **n**.

## Practicalities

- Budget about two minutes per answer item and under a minute per refusal.
- Do the dev split first if you want early feedback; calibration reports dev disagreements
  in detail and held-out ones only as a count.
- Labels are by one annotator. If someone else labels too, pass `--annotator NAME`; the
  report does not yet compute inter-annotator agreement.
