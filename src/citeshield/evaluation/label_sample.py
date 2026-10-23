"""Build the frozen labelling sample for judge calibration.

    uv run citeshield-label-sample          # writes eval/labels/sample.jsonl; never overwrites
    uv run citeshield-label-sample --force

Deterministic for a given seed and set of stored runs (``eval/runs``, gitignored). The
written file is the source of truth: labels refer to its items, so it is committed and
never regenerated silently.

Composition (``Quotas``):

* real answers: answered or partial answers to answerable questions, at most one per
  identical answer text, with the current golden reference attached. One per question
  first, partial answers always in, then further answers to the same questions.
* synthetic perturbations of real answers, made deterministically and marked
  ``origin="synthetic"``: a claim strengthened beyond its source, a citation swapped to an
  unrelated retrieved provision, a fact contradicted (a number changed, or a modal negated).
  They exist because the verifier removes most unsupported claims and invalid citations
  before an answer is final, so real answers alone would give almost no negative examples.
  A person still labels every one; nothing about them is assumed.
* refusals: appropriate (a question that must be refused) and incorrect (an answerable
  question that was refused), one per question.

Split: by golden question id, so every item of a question (several answers, perturbations)
lands in the same half. Questions are stratified by their rarest outcome and assigned
greedily to keep both halves balanced per stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from citeshield.config import PROJECT_ROOT, Settings
from citeshield.contracts.chunk import ChunkRecord
from citeshield.evaluation.annotations import SAMPLE_PATH, LabelItem, Outcome, Split
from citeshield.evaluation.golden import GoldenItem, load_golden
from citeshield.evaluation.records import AnswerRecord, CitedSource, load_records

RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden" / "questions.jsonl"
SEED = "citeshield-labels-v1"

Lookup = Callable[[str], ChunkRecord | None]

# Rarest first: a question's stratum is the first of these among its items.
STRATA: tuple[Outcome, ...] = (
    "perturbed_wrong_citation",
    "perturbed_contradiction",
    "perturbed_unsupported",
    "partial",
    "wrong_refusal",
    "correct_refusal",
    "answered",
)

UNSUPPORTED_ADDITIONS = (
    " This applies to every organisation, without exception.",
    " It must also be repeated every twelve months.",
    " Failing to do so is a criminal offence under this Regulation.",
    " This must be done within 24 hours.",
)
# A quantity with its unit ("72 hours", "4 %", "EUR 20 000 000" is left alone: spaced
# digits), or a year. Article, paragraph and point numbers are cross-references, not facts,
# and are never changed.
_QUANTITY = re.compile(
    r"(?<![\w.(])(\d+)(\s*(?:hours?|days?|weeks?|months?|years?|%|per ?cent|million))\b",
    re.IGNORECASE,
)
# A year, but not the year part of an act number such as "Regulation (EU) 2016/679".
_YEAR = re.compile(r"(?<![\w.(/])((?:19|20)\d{2})(?![\w)/])")
_MODALS = ((" shall not ", " shall "), (" must not ", " must "), (" may not ", " may "))


@dataclass(frozen=True)
class Quotas:
    real: int = 36
    per_perturbation: int = 4
    correct_refusals: int = 6
    wrong_refusals: int = 6


def _h(*parts: str) -> str:
    return hashlib.sha256(":".join((SEED, *parts)).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    run: str
    record: AnswerRecord


def collect(runs_dir: Path, golden: dict[str, GoldenItem]) -> list[Candidate]:
    """Every distinct stored answer with the current golden text attached.

    Runs that recorded runtime metadata (current eval code) come first, so when the same
    answer text exists twice the newer run is named as its source.
    """
    runs = sorted(p for p in runs_dir.iterdir() if (p / "answers.jsonl").exists())

    def current(run: Path) -> bool:
        meta = run / "meta.json"
        return meta.exists() and "runtime" in json.loads(meta.read_text("utf-8"))

    seen: set[tuple[str, str]] = set()
    out: list[Candidate] = []
    for run in sorted(runs, key=lambda r: (not current(r), r.name)):
        for rec in load_records(run / "answers.jsonl"):
            item = golden.get(rec.id)
            if item is None or rec.crashed:
                continue
            rec = rec.model_copy(
                update={
                    "question": item.question,
                    "expected": item.expected,
                    "should_refuse": item.should_refuse,
                    "reference_answer": item.reference_answer,
                }
            )
            key = (rec.id, rec.view_sha)
            if key not in seen:
                seen.add(key)
                out.append(Candidate(run.name, rec))
    return out


# ------------------------------------------------------------------------ perturbations


def _with_answer(rec: AnswerRecord, answer: dict[str, object], **update: object) -> AnswerRecord:
    from citeshield.contracts.answer import AgentAnswer

    return rec.model_copy(update={"answer": AgentAnswer.model_validate(answer), **update})


def perturb_unsupported(rec: AnswerRecord, k: int) -> tuple[AnswerRecord, str] | None:
    assert rec.answer is not None
    answer = rec.answer.model_dump(mode="json")
    if not answer["claims"]:
        return None
    addition = UNSUPPORTED_ADDITIONS[k % len(UNSUPPORTED_ADDITIONS)]
    claim = answer["claims"][0]
    if len(claim["text"]) + len(addition) > 1000:
        return None
    claim["text"] = claim["text"].rstrip() + addition
    return _with_answer(rec, answer), f"claim 1: appended {addition.strip()!r}"


def perturb_contradiction(rec: AnswerRecord) -> tuple[AnswerRecord, str] | None:
    assert rec.answer is not None
    answer = rec.answer.model_dump(mode="json")
    changes: tuple[tuple[re.Pattern[str], Callable[[int], int]], ...] = (
        (_QUANTITY, lambda n: n * 2),  # a quantity doubles
        (_YEAR, lambda n: n + 1),  # a year moves by one
    )
    for regex, change in changes:
        for i, claim in enumerate(answer["claims"]):
            m = regex.search(claim["text"])
            if m:
                old = m.group(0)
                new = str(change(int(m.group(1)))) + old[len(m.group(1)) :]
                claim["text"] = claim["text"].replace(old, new)
                answer["summary"] = answer["summary"].replace(old, new)
                return _with_answer(rec, answer), f"claim {i + 1} and summary: {old!r} -> {new!r}"
    for i, claim in enumerate(answer["claims"]):
        padded = f" {claim['text']} "
        for negated, plain in _MODALS:
            if plain in padded and negated not in padded:
                claim["text"] = padded.replace(plain, negated, 1).strip()
                return _with_answer(rec, answer), f"claim {i + 1}: {plain.strip()!r} negated"
    return None


def perturb_wrong_citation(rec: AnswerRecord, lookup: Lookup) -> tuple[AnswerRecord, str] | None:
    assert rec.answer is not None
    answer = rec.answer.model_dump(mode="json")
    if not answer["claims"]:
        return None
    refs = [r for r in rec.expected]
    from citeshield.evaluation.golden import parse_ref

    parsed = [parse_ref(r) for r in refs]
    cited = set(rec.cited)
    options = []
    for cid in rec.retrieved:
        chunk = lookup(cid)
        if chunk is None or cid in cited or chunk.kind.value != "article":
            continue
        if any(r.matches(chunk) for r in parsed):
            continue
        options.append(chunk)
    if not options:
        return None
    chunk = min(options, key=lambda c: _h(rec.id, rec.view_sha, c.chunk_id))
    claim = answer["claims"][0]
    old_id = claim["citations"][0]["chunk_id"]
    quote = re.sub(r"^\(?\w{1,4}[.)]\s*", "", chunk.text).strip()[:200]
    if len(quote) < 8:
        return None
    claim["citations"][0] = {"chunk_id": chunk.chunk_id, "quote": quote}
    still_cited = {c["chunk_id"] for cl in answer["claims"] for c in cl["citations"]}
    still_cited |= {c["chunk_id"] for it in answer.get("checklist") or [] for c in it["citations"]}
    sources = {k: v for k, v in rec.cited.items() if k in still_cited}
    sources[chunk.chunk_id] = CitedSource(label=chunk.citation_label, text=chunk.text)
    new = _with_answer(rec, answer, cited=sources)
    label = rec.cited[old_id].label if old_id in rec.cited else old_id
    return new, f"claim 1: citation {label} replaced by non-expected {chunk.citation_label}"


# ------------------------------------------------------------------------ selection


def _real_outcome(rec: AnswerRecord) -> Outcome:
    assert rec.answer is not None
    return "partial" if rec.answer.status.value == "partial" else "answered"


def _pick_spread(
    cands: Sequence[Candidate], n: int, forced: Sequence[Candidate] = ()
) -> list[Candidate]:
    """One per question first (hash order), then more answers to the same questions."""
    chosen = list(forced)[:n]
    rest = sorted(
        (c for c in cands if c not in chosen), key=lambda c: _h(c.record.id, c.record.view_sha)
    )
    have = {c.record.id for c in chosen}
    for c in rest:
        if len(chosen) < n and c.record.id not in have:
            chosen.append(c)
            have.add(c.record.id)
    for c in rest:
        if len(chosen) < n and c not in chosen:
            chosen.append(c)
    return chosen


def _one_per_question(cands: Sequence[Candidate], n: int) -> list[Candidate]:
    picked: list[Candidate] = []
    have: set[str] = set()
    for c in sorted(cands, key=lambda c: _h(c.record.id, c.record.view_sha)):
        if len(picked) < n and c.record.id not in have:
            picked.append(c)
            have.add(c.record.id)
    return picked


def assign_splits(items: Sequence[tuple[str, Outcome]]) -> dict[str, Split]:
    """Question-level split: (question_id, outcome) pairs -> split per question."""
    outcomes: dict[str, set[Outcome]] = {}
    for qid, outcome in items:
        outcomes.setdefault(qid, set()).add(outcome)
    size = {qid: sum(1 for q, _ in items if q == qid) for qid in outcomes}
    stratum = {qid: next(s for s in STRATA if s in outs) for qid, outs in outcomes.items()}
    counts: dict[tuple[Outcome, Split], int] = {}
    totals: dict[Split, int] = {"dev": 0, "holdout": 0}
    split: dict[str, Split] = {}
    for qid in sorted(outcomes, key=lambda q: (STRATA.index(stratum[q]), _h("split", q))):
        s = stratum[qid]

        def load(sp: Split, s: Outcome = s) -> tuple[int, int, int]:
            return (counts.get((s, sp), 0), totals[sp], 0 if sp == "dev" else 1)

        target: Split = min(("dev", "holdout"), key=load)
        split[qid] = target
        counts[(s, target)] = counts.get((s, target), 0) + size[qid]
        totals[target] += size[qid]
    return split


def build_sample(
    cands: Sequence[Candidate], lookup: Lookup, quotas: Quotas | None = None
) -> tuple[list[LabelItem], list[str]]:
    """The sample and any shortfalls against the quotas."""
    quotas = quotas or Quotas()
    answers = [c for c in cands if not c.record.refused and not c.record.should_refuse
               and c.record.reference_answer is not None]  # fmt: skip
    partial = [c for c in answers if _real_outcome(c.record) == "partial"]
    real = _pick_spread(answers, quotas.real, forced=partial)
    correct = _one_per_question([c for c in cands if c.record.refused and c.record.should_refuse],
                                quotas.correct_refusals)  # fmt: skip
    wrong = _one_per_question([c for c in cands if c.record.refused and not c.record.should_refuse],
                              quotas.wrong_refusals)  # fmt: skip

    # Bases for perturbation: answers with claims, leftovers before already-chosen ones.
    bases = sorted(
        (c for c in answers if c.record.answer and c.record.answer.claims),
        key=lambda c: (c in real, _h("base", c.record.id, c.record.view_sha)),
    )
    used: set[tuple[str, str]] = set()
    synthetic: list[tuple[Candidate, AnswerRecord, Outcome, str]] = []
    makers: list[tuple[Outcome, Callable[[AnswerRecord, int], tuple[AnswerRecord, str] | None]]] = [
        ("perturbed_unsupported", perturb_unsupported),
        ("perturbed_wrong_citation", lambda r, _k: perturb_wrong_citation(r, lookup)),
        ("perturbed_contradiction", lambda r, _k: perturb_contradiction(r)),
    ]
    shortfalls: list[str] = []
    perturbed_questions: set[str] = set()
    for outcome, make in makers:
        made = 0
        # First pass: questions not perturbed yet; second pass: any unused base.
        for fresh_only in (True, False):
            for c in bases:
                key = (c.record.id, c.record.view_sha)
                if made == quotas.per_perturbation:
                    break
                if key in used or (fresh_only and c.record.id in perturbed_questions):
                    continue
                result = make(c.record, made)
                if result is None:
                    continue
                used.add(key)
                perturbed_questions.add(c.record.id)
                synthetic.append((c, result[0], outcome, result[1]))
                made += 1
        if made < quotas.per_perturbation:
            shortfalls.append(f"{outcome}: {made} of {quotas.per_perturbation}")
    for name, got, want in (("real answers", len(real), quotas.real),
                            ("correct refusals", len(correct), quotas.correct_refusals),
                            ("wrong refusals", len(wrong), quotas.wrong_refusals)):  # fmt: skip
        if got < want:
            shortfalls.append(f"{name}: {got} of {want}")

    rows: list[tuple[Candidate, AnswerRecord, Outcome, str | None]] = [
        *((c, c.record, _real_outcome(c.record), None) for c in real),
        *((c, c.record, "correct_refusal", None) for c in correct),
        *((c, c.record, "wrong_refusal", None) for c in wrong),
        *synthetic,
    ]
    splits = assign_splits([(rec.id, outcome) for _, rec, outcome, _ in rows])
    items = [
        LabelItem(
            item_id=f"{rec.id}-{rec.view_sha}",
            question_id=rec.id,
            kind="refusal" if rec.refused else "answer",
            origin="synthetic" if pert else "real",
            outcome=outcome,
            split=splits[rec.id],
            source_run=c.run,
            perturbation=pert,
            record=rec,
            view_sha=rec.view_sha,
        )
        for c, rec, outcome, pert in rows
    ]
    return sorted(items, key=lambda i: i.item_id), shortfalls


def summary(items: Sequence[LabelItem]) -> str:
    lines = ["| outcome | dev | holdout |", "|---|---|---|"]
    for o in STRATA:
        d = sum(1 for i in items if i.outcome == o and i.split == "dev")
        h = sum(1 for i in items if i.outcome == o and i.split == "holdout")
        lines.append(f"| {o} | {d} | {h} |")
    d = sum(1 for i in items if i.split == "dev")
    lines.append(f"| total | {d} | {len(items) - d} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--runs", type=Path, default=RUNS_DIR)
    ap.add_argument("--force", action="store_true", help="Overwrite an existing sample.")
    args = ap.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} exists; labels refer to it. Use --force to rebuild it.")
        return 1
    from citeshield.retrieval.corpus import Corpus

    golden = {i.id: i for i in load_golden(GOLDEN_PATH)}
    corpus = Corpus.load(Settings().chunks_path)
    items, shortfalls = build_sample(collect(args.runs, golden), corpus.get)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(i.model_dump_json() + "\n" for i in items), "utf-8", newline="\n")
    print(summary(items))
    for s in shortfalls:
        print(f"shortfall: {s}")
    print(f"wrote {len(items)} items to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
