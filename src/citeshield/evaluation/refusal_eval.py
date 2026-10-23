"""Deterministic over-refusal evaluation of stored answer-eval runs (no LLM, no judge).

Ground truth is ``should_refuse`` and ``category`` from the golden set
(``eval/golden/questions.jsonl``). The copy each record carries is only cross-checked: a
record whose labels disagree with the golden set, or whose id is not in it, stops the
evaluation instead of being scored. Labels are never inferred from what the agent did.

Every record gets exactly one outcome:

* ``answered`` / ``partial``: a final answer with that status. Both are substantive (they
  contain cited claims or checklist items).
* ``refused``: a deliberate refusal (guard, provider, drafting model, retrieval threshold or
  verifier; see ``RefusalSource``).
* ``invalid_output``: the structured output never became a valid answer: the final answer
  does not round-trip the contract, the run died on a parse/validation error, or the draft
  stayed malformed after repair and ``graph.salvage`` fell back to a refusal (see
  ``salvaged``; ``policy.finalize`` drops the salvage caveat, so the caveat is no signal).
* ``timeout``: the run raised a timeout (``ReadTimeout``, ``TimeoutError``, ...).
* ``crashed``: any other exception, or a record with neither answer nor error.

The last three are failures and are never counted as refusals.

Rates (each reported with its count and denominator; ``None`` when the denominator is 0):

* over-refusal rate = gold-answerable refused / gold-answerable
* answer coverage = gold-answerable answered or partial / gold-answerable
* full-answer rate, partial rate = answered (resp. partial) / gold-answerable
* answerable failure rate = gold-answerable invalid_output, timeout or crashed / gold-answerable
* over-refusal rate (completed) = gold-answerable refused / gold-answerable not failed
* correct refusal rate = gold-unanswerable refused / gold-unanswerable
* under-refusal rate = gold-unanswerable answered or partial / gold-unanswerable
* under-refusal partial rate = gold-unanswerable partial / gold-unanswerable
* unanswerable failure rate = gold-unanswerable failed / gold-unanswerable

Denominators include failed items, so over-refusal + coverage + failure rate = 1 for the
answerable class. A correct refusal counts any refusal reason; whether the reason is the
right one is not checked (the golden set records no expected reason).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from citeshield.contracts.answer import AnswerStatus, RefusalReason
from citeshield.evaluation.golden import Category, GoldenItem, load_golden
from citeshield.evaluation.records import AnswerRecord, load_records
from citeshield.evaluation.run_eval import GOLDEN_PATH, golden_sha


class Outcome(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    REFUSED = "refused"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    CRASHED = "crashed"


SUBSTANTIVE = frozenset({Outcome.ANSWERED, Outcome.PARTIAL})
FAILURES = frozenset({Outcome.INVALID_OUTPUT, Outcome.TIMEOUT, Outcome.CRASHED})


class RefusalSource(StrEnum):
    GUARD = "guard"
    PROVIDER = "provider"  # the LLM API declined (LLMRefusal)
    DRAFT = "draft"  # the drafting model chose to refuse
    RETRIEVAL_THRESHOLD = "retrieval_threshold"  # policy.weak_evidence
    VERIFIER = "verifier"  # every claim failed verification
    UNKNOWN = "unknown"


# Exception names (the "ExcType" in ``AnswerRecord.error``).
TIMEOUT_ERRORS = frozenset(
    {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
     "TimeoutError"}
)  # fmt: skip
INVALID_OUTPUT_ERRORS = frozenset({"ValidationError", "JSONDecodeError"})

# Caveat prefixes written by the code paths that produce each refusal. Pinned by tests
# against agent/graph.py and agent/policy.py, so a wording change there fails loudly.
PROVIDER_CAVEAT = "model declined the request"
RETRIEVAL_CAVEAT = "Best retrieval score"
VERIFIER_CAVEAT = "Removed "


def _error_type(error: str | None) -> str:
    return (error or "").split(":", 1)[0].strip()


def _has_caveat(rec: AnswerRecord, prefix: str) -> bool:
    return rec.answer is not None and any(c.startswith(prefix) for c in rec.answer.caveats)


def salvaged(rec: AnswerRecord) -> bool:
    """Whether the final answer came from the salvage fallback.

    Records from before M7 phase 2 have no ``salvaged`` flag. With ``max_draft_attempts=2``
    (the only setting any stored run used) the second rejected draft always triggers
    salvage, so two stored rejections imply it. Records without ``draft_rejections``
    (before M7) cannot tell, and count as not salvaged.
    """
    if rec.salvaged is not None:
        return rec.salvaged
    return len(rec.draft_rejections) >= 2


def outcome(rec: AnswerRecord) -> Outcome:
    if rec.answer is None:
        kind = _error_type(rec.error)
        if kind in TIMEOUT_ERRORS:
            return Outcome.TIMEOUT
        if kind in INVALID_OUTPUT_ERRORS:
            return Outcome.INVALID_OUTPUT
        return Outcome.CRASHED
    if not rec.schema_valid:
        return Outcome.INVALID_OUTPUT
    if rec.answer.status is AnswerStatus.REFUSED:
        return Outcome.INVALID_OUTPUT if salvaged(rec) else Outcome.REFUSED
    if rec.answer.status is AnswerStatus.PARTIAL:
        return Outcome.PARTIAL
    return Outcome.ANSWERED


def refusal_source(rec: AnswerRecord) -> RefusalSource | None:
    """Which stage refused, read from the stored record; None if it is not a refusal."""
    if outcome(rec) is not Outcome.REFUSED:
        return None
    assert rec.answer is not None
    if rec.guard_label is not None and rec.guard_label != "in_scope":
        return RefusalSource.GUARD
    if _has_caveat(rec, PROVIDER_CAVEAT):
        return RefusalSource.PROVIDER
    if _has_caveat(rec, RETRIEVAL_CAVEAT):
        return RefusalSource.RETRIEVAL_THRESHOLD
    if _has_caveat(rec, VERIFIER_CAVEAT):
        return RefusalSource.VERIFIER
    # policy.finalize drops a drafted refusal's caveats and keeps only its reason.
    if not rec.answer.caveats and rec.guard_label == "in_scope":
        return RefusalSource.DRAFT
    return RefusalSource.UNKNOWN


# ------------------------------------------------------------------------ labels


class LabelError(ValueError):
    """Records that cannot be matched to trusted golden labels."""


@dataclass(frozen=True)
class Scored:
    id: str
    category: Category
    should_refuse: bool  # from the golden set
    outcome: Outcome
    refusal_reason: RefusalReason | None
    refusal_source: RefusalSource | None
    first_draft_valid: bool


def score(records: Sequence[AnswerRecord], golden: Sequence[GoldenItem]) -> list[Scored]:
    by_id = {g.id: g for g in golden}
    problems = []
    ids = Counter(r.id for r in records)
    problems += [f"{i}: {n} records" for i, n in sorted(ids.items()) if n > 1]
    for r in records:
        g = by_id.get(r.id)
        if g is None:
            problems.append(f"{r.id}: not in the golden set")
        elif (r.should_refuse, r.category) != (g.should_refuse, g.category):
            problems.append(
                f"{r.id}: record says should_refuse={r.should_refuse}/{r.category.value}, "
                f"golden says {g.should_refuse}/{g.category.value}"
            )
    if problems:
        raise LabelError("; ".join(problems))
    return [
        Scored(
            id=r.id,
            category=by_id[r.id].category,
            should_refuse=by_id[r.id].should_refuse,
            outcome=outcome(r),
            refusal_reason=r.answer.refusal_reason if r.answer else None,
            refusal_source=refusal_source(r),
            first_draft_valid=r.first_draft_valid,
        )
        for r in records
    ]


# ------------------------------------------------------------------------ metrics


@dataclass(frozen=True)
class Rate:
    count: int
    denominator: int

    @property
    def value(self) -> float | None:
        return self.count / self.denominator if self.denominator else None

    def __str__(self) -> str:
        v = self.value
        return f"{'n/a' if v is None else f'{v:.3f}'} ({self.count}/{self.denominator})"


def _rate(items: Sequence[Scored], hit: Iterable[Outcome]) -> Rate:
    wanted = set(hit)
    return Rate(sum(s.outcome in wanted for s in items), len(items))


def rates(items: Sequence[Scored]) -> dict[str, Rate]:
    ans = [s for s in items if not s.should_refuse]
    una = [s for s in items if s.should_refuse]
    ans_done = [s for s in ans if s.outcome not in FAILURES]
    return {
        "over_refusal_rate": _rate(ans, [Outcome.REFUSED]),
        "answer_coverage": _rate(ans, SUBSTANTIVE),
        "full_answer_rate": _rate(ans, [Outcome.ANSWERED]),
        "partial_answer_rate": _rate(ans, [Outcome.PARTIAL]),
        "answerable_failure_rate": _rate(ans, FAILURES),
        "over_refusal_rate_completed": _rate(ans_done, [Outcome.REFUSED]),
        "correct_refusal_rate": _rate(una, [Outcome.REFUSED]),
        "under_refusal_rate": _rate(una, SUBSTANTIVE),
        "under_refusal_partial_rate": _rate(una, [Outcome.PARTIAL]),
        "unanswerable_failure_rate": _rate(una, FAILURES),
    }


def outcome_counts(items: Sequence[Scored]) -> dict[str, dict[str, int]]:
    """Outcome counts per gold class; every outcome is listed, zeros included."""
    out = {}
    for name, flag in (("answerable", False), ("unanswerable", True)):
        c = Counter(s.outcome for s in items if s.should_refuse is flag)
        out[name] = {o.value: c[o] for o in Outcome}
    return out


def by_category(items: Sequence[Scored]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cat in sorted({s.category for s in items}):
        group = [s for s in items if s.category is cat]
        c = Counter(s.outcome for s in group)
        out[cat.value] = {
            "n": len(group),
            "gold": "unanswerable" if all(s.should_refuse for s in group) else (
                "answerable" if not any(s.should_refuse for s in group) else "mixed"),
            **{o.value: c[o] for o in Outcome},
            "refused": _rate(group, [Outcome.REFUSED]),
            "substantive": _rate(group, SUBSTANTIVE),
        }  # fmt: skip
    return out


def over_refusal_breakdown(items: Sequence[Scored]) -> dict[str, Counter[str]]:
    """Refusals of gold-answerable questions by refusing stage and by refusal reason.

    ``after_rejected_draft``: the same stages, counting only runs whose first draft failed
    validation (a refusal written during repair shows up here as ``draft``).
    """
    refused = [s for s in items if not s.should_refuse and s.outcome is Outcome.REFUSED]
    return {
        "source": Counter(s.refusal_source.value for s in refused if s.refusal_source),
        "reason": Counter(s.refusal_reason.value for s in refused if s.refusal_reason),
        "after_rejected_draft": Counter(
            s.refusal_source.value for s in refused if s.refusal_source and not s.first_draft_valid
        ),
    }


# ------------------------------------------------------------------------ runs


def provenance(
    run_dir: Path, records: Sequence[AnswerRecord], golden: Sequence[GoldenItem]
) -> dict[str, Any]:
    """Run identity, and whether the stored run still matches the golden set it used."""
    answers = run_dir / "answers.jsonl"
    meta_path = run_dir / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    by_id = {g.id: g for g in golden}
    run_ids = [r.id for r in records]
    planned = meta.get("ids")
    agent = meta.get("agent", {})
    code = meta.get("runtime", {}).get("code", {})
    # The recorded hash covers the planned ids, so hash the same ids.
    current = golden_sha([by_id[i] for i in (planned or run_ids) if i in by_id])
    return {
        "run_dir": run_dir.as_posix(),
        "answers_sha256": hashlib.sha256(answers.read_bytes()).hexdigest()[:16],
        "has_meta": bool(meta),
        "run_id": meta.get("run_id"),
        "subset": meta.get("subset"),
        "provider": agent.get("llm_provider"),
        "agent_model": agent.get("agent_model"),
        "prompt_versions": agent.get("prompt_versions"),
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "n_records": len(records),
        "n_planned": len(planned) if planned is not None else None,
        "not_run": sorted(set(planned) - set(run_ids)) if planned is not None else None,
        "golden_sha_recorded": meta.get("golden_sha"),
        "golden_sha_current": current,
        "golden_unchanged": meta.get("golden_sha") == current if meta.get("golden_sha") else None,
    }


def evaluate_run(
    run_dir: Path, golden: Sequence[GoldenItem], *, reviewed_only: bool = False
) -> dict[str, Any]:
    """``reviewed_only`` scores only items whose golden entry a human has reviewed."""
    records = load_records(run_dir / "answers.jsonl")
    items = score(records, golden)  # labels are checked on every record either way
    if reviewed_only:
        reviewed = {g.id for g in golden if g.reviewed}
        items = [s for s in items if s.id in reviewed]
    return {
        "provenance": provenance(run_dir, records, golden),
        "scope": f"human-reviewed golden items only ({len(items)} of {len(records)})"
        if reviewed_only
        else f"all records ({len(items)})",
        "outcomes": outcome_counts(items),
        "rates": rates(items),
        "by_category": by_category(items),
        "over_refusals": over_refusal_breakdown(items),
        "over_refused_ids": sorted(
            s.id for s in items if not s.should_refuse and s.outcome is Outcome.REFUSED
        ),
    }


def render(result: dict[str, Any]) -> str:
    p = result["provenance"]
    planned = "unknown" if p["n_planned"] is None else p["n_planned"]
    lines = [
        f"## `{p['run_id'] or p['run_dir']}`",
        "",
        f"- records {p['n_records']} (planned {planned}"
        f"{', not run: ' + ', '.join(p['not_run']) if p['not_run'] else ''}); "
        f"subset `{p['subset']}`; provider `{p['provider']}` / `{p['agent_model']}`",
        f"- answers.jsonl sha256 `{p['answers_sha256']}`; meta.json "
        f"{'present' if p['has_meta'] else '**missing** (no run identity)'}; "
        f"git `{(p['git_commit'] or 'unknown')[:7]}` dirty={p['git_dirty']}",
        f"- golden sha recorded `{p['golden_sha_recorded']}`, current `{p['golden_sha_current']}`"
        f" (unchanged: {p['golden_unchanged']})",
        f"- scored: {result['scope']}",
        "",
        "| gold class | " + " | ".join(o.value for o in Outcome) + " | total |",
        "|---|" + "---|" * (len(Outcome) + 1),
    ]
    for cls, counts in result["outcomes"].items():
        lines.append(f"| {cls} | " + " | ".join(str(counts[o.value]) for o in Outcome)
                     + f" | {sum(counts.values())} |")  # fmt: skip
    lines += ["", "| metric | value (count/denominator) |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in result["rates"].items()]
    lines += [
        "",
        "| category | gold | n | answered | partial | refused | failed | refused share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cat, row in result["by_category"].items():
        failed = sum(row[o.value] for o in FAILURES)
        lines.append(
            f"| {cat} | {row['gold']} | {row['n']} | {row['answered']} | {row['partial']} "
            f"| {row['refused'].count} | {failed} | {row['refused']} |"
        )
    o = result["over_refusals"]
    lines += [
        "",
        f"Over-refusals by stage: {dict(o['source'].most_common()) or 'none'}; "
        f"by reason: {dict(o['reason'].most_common()) or 'none'}; "
        f"after a rejected first draft: {dict(o['after_rejected_draft'].most_common()) or 'none'}.",
        f"Over-refused ids: {', '.join(result['over_refused_ids']) or 'none'}.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="eval/runs/<run_id> directories")
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH, help="Trusted labels.")
    ap.add_argument("--out", type=Path, help="Also write the markdown here.")
    ap.add_argument(
        "--reviewed-only", action="store_true", help="Score only human-reviewed golden items."
    )
    ap.add_argument("--mlflow", action="store_true", help="Also log each run to MLflow.")
    args = ap.parse_args(argv)

    golden = load_golden(args.golden)
    parts = []
    for run_dir in args.runs:
        try:
            result = evaluate_run(run_dir, golden, reviewed_only=args.reviewed_only)
            parts.append(render(result))
            if args.mlflow:
                _log_mlflow(run_dir, result)
        except LabelError as e:
            print(f"{run_dir}: cannot score against trusted labels: {e}", file=sys.stderr)
            return 2
    text = "\n\n".join(parts) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


def _log_mlflow(run_dir: Path, result: dict[str, Any]) -> None:
    from citeshield.observability import log_eval_run

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
    meta.setdefault("run_id", run_dir.name)
    metrics: dict[str, Any] = {k: r.value for k, r in result["rates"].items()}
    metrics |= {f"{k}_count": r.count for k, r in result["rates"].items()}
    metrics |= {f"{k}_n": r.denominator for k, r in result["rates"].items()}
    log_eval_run(meta, metrics, kind="refusal-eval",
                 artifacts={"outcomes.json": result["outcomes"]})  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
