"""Deterministic over-refusal evaluation: outcomes, refusal stages, rates, labels, CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from citeshield.agent.policy import finalize
from citeshield.contracts.answer import AgentAnswer
from citeshield.evaluation import refusal_eval as rv
from citeshield.evaluation.golden import GoldenItem
from citeshield.evaluation.records import AnswerRecord
from citeshield.retrieval.corpus import Corpus
from citeshield.verify.verifier import Verifier
from helpers import RuleNLI
from vfx import EXEMPT, EXEMPT_QUOTE, NEUTRAL, P2, PTA, answer, evidence

Outcome = rv.Outcome
Source = rv.RefusalSource

CLAIM = {"text": "Claim.", "citations": [{"chunk_id": "c1", "quote": "shall be lawful"}]}
ANSWERED = AgentAnswer.model_validate({"status": "answered", "summary": "S.", "claims": [CLAIM]})
PARTIAL = AgentAnswer.model_validate(
    {
        "status": "partial",
        "summary": "S.",
        "claims": [CLAIM],
        "refusal_reason": "unsupported_claims",
    }
)


def refused(reason: str = "insufficient_evidence", caveats: tuple[str, ...] = ()) -> AgentAnswer:
    return AgentAnswer.model_validate(
        {"status": "refused", "summary": "No.", "refusal_reason": reason, "caveats": list(caveats)}
    )


def gold(rid: str, *, should_refuse: bool = False, category: str | None = None) -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "id": rid,
            "question": "Q?",
            "category": category or ("unanswerable" if should_refuse else "lookup"),
            "expected": [] if should_refuse else ["GDPR 6(1)"],
            "should_refuse": should_refuse,
            "reference_answer": None if should_refuse else "Ref.",
        }
    )


def record(
    g: GoldenItem,
    answer: AgentAnswer | None = None,
    *,
    error: str | None = None,
    guard_label: str | None = "in_scope",
    **kw: Any,
) -> AnswerRecord:
    return AnswerRecord.model_validate(
        {
            "id": g.id,
            "category": g.category,
            "question": g.question,
            "should_refuse": g.should_refuse,
            "expected": g.expected,
            "reference_answer": g.reference_answer,
            "answer": answer,
            "error": error,
            "guard_label": guard_label if answer is not None else None,
            "schema_valid": answer is not None,
            **kw,
        }
    )


A = gold("q001")
U = gold("q002", should_refuse=True)

# ------------------------------------------------------------------------ outcomes


@pytest.mark.parametrize(
    ("rec", "expected"),
    [
        (record(A, ANSWERED), Outcome.ANSWERED),
        (record(A, PARTIAL), Outcome.PARTIAL),
        (record(A, refused()), Outcome.REFUSED),
        (record(A, error="LLMContextExceeded: prompt is ~9000 tokens"), Outcome.CRASHED),
        (record(A, error="ReadTimeout: timed out"), Outcome.TIMEOUT),
        (record(A, error="TimeoutError: no reply"), Outcome.TIMEOUT),
        (record(A, error="ValidationError: 1 validation error"), Outcome.INVALID_OUTPUT),
        (record(A, error="JSONDecodeError: Expecting value"), Outcome.INVALID_OUTPUT),
        (record(A), Outcome.CRASHED),  # neither answer nor error: a broken record
        (record(A, ANSWERED, schema_valid=False), Outcome.INVALID_OUTPUT),
    ],
)
def test_outcome(rec: AnswerRecord, expected: rv.Outcome) -> None:
    assert rv.outcome(rec) is expected


def test_salvage_fallback_refusal_is_invalid_output_not_refusal() -> None:
    # What a salvage refusal looks like after policy.finalize: a plain refusal, no caveats.
    rec = record(A, refused(), salvaged=True)
    assert rv.outcome(rec) is Outcome.INVALID_OUTPUT
    assert rv.refusal_source(rec) is None
    # A salvaged partial answer still has cited claims: it stays substantive.
    assert rv.outcome(record(A, PARTIAL, salvaged=True)) is Outcome.PARTIAL
    # A deliberate refusal written during a successful repair is not salvage.
    assert rv.outcome(record(A, refused(), salvaged=False, draft_rejections=[["e"]])) is (
        Outcome.REFUSED
    )


@pytest.mark.parametrize(
    ("rejections", "expected"),
    [
        ([], Outcome.REFUSED),  # before M7: no rejection details, cannot tell
        ([["quote too long"]], Outcome.REFUSED),  # one rejection, then a valid refusal
        ([["quote too long"], ["quote too long"]], Outcome.INVALID_OUTPUT),  # budget spent
    ],
)
def test_salvage_inferred_for_records_without_the_flag(
    rejections: list[list[str]], expected: rv.Outcome
) -> None:
    rec = record(A, refused(), draft_rejections=rejections)
    assert rec.salvaged is None
    assert rv.outcome(rec) is expected


# ------------------------------------------------------------------------ refusal stages


def _finalized(corpus: Corpus, draft: AgentAnswer, nli: RuleNLI, score: float) -> AgentAnswer:
    return finalize(
        draft, Verifier(corpus, nli).verify(draft, evidence(corpus, P2, PTA, score=score)), 0.30
    )


def test_refusal_sources_match_the_policy_that_produced_them(corpus: Corpus) -> None:
    good = answer((EXEMPT, [(P2, EXEMPT_QUOTE)]))
    weak = _finalized(corpus, good, RuleNLI({EXEMPT: NEUTRAL}), score=0.1)
    unsupported = _finalized(corpus, good, RuleNLI({EXEMPT: NEUTRAL}), score=0.9)
    drafted = _finalized(corpus, refused("out_of_scope", ("model prose",)), RuleNLI({}), 0.9)

    assert rv.refusal_source(record(A, weak)) is Source.RETRIEVAL_THRESHOLD
    assert rv.refusal_source(record(A, unsupported)) is Source.VERIFIER
    assert rv.refusal_source(record(A, drafted)) is Source.DRAFT


def test_guard_provider_and_unknown_sources() -> None:
    assert rv.refusal_source(record(U, refused("out_of_scope"), guard_label="other_law")) is (
        Source.GUARD
    )
    provider = refused("out_of_scope", ("model declined the request (category=cyber)",))
    assert rv.refusal_source(record(A, provider)) is Source.PROVIDER
    # A guard label missing from an old record: the stage cannot be told apart.
    assert rv.refusal_source(record(A, refused(), guard_label=None)) is Source.UNKNOWN
    assert rv.refusal_source(record(A, ANSWERED)) is None


# ------------------------------------------------------------------------ rates


def test_rates_counts_and_denominators() -> None:
    ans = [gold(f"q{i:03d}") for i in range(1, 9)]
    una = [gold(f"q{i:03d}", should_refuse=True) for i in range(11, 15)]
    recs = [
        record(ans[0], ANSWERED),
        record(ans[1], ANSWERED),
        record(ans[2], PARTIAL),
        record(ans[3], refused()),
        record(ans[4], refused()),
        record(ans[5], error="LLMContextExceeded: too long"),
        record(ans[6], error="ReadTimeout: timed out"),
        record(ans[7], refused(), salvaged=True),
        record(una[0], refused("out_of_scope"), guard_label="other_law"),
        record(una[1], refused("injection_detected"), guard_label="manipulation"),
        record(una[2], PARTIAL),
        record(una[3], error="RuntimeError: boom"),
    ]
    items = rv.score(recs, ans + una)
    r = {k: (v.count, v.denominator) for k, v in rv.rates(items).items()}
    assert r == {
        "over_refusal_rate": (2, 8),
        "answer_coverage": (3, 8),
        "full_answer_rate": (2, 8),
        "partial_answer_rate": (1, 8),
        "answerable_failure_rate": (3, 8),
        "over_refusal_rate_completed": (2, 5),
        "correct_refusal_rate": (2, 4),
        "under_refusal_rate": (1, 4),
        "under_refusal_partial_rate": (1, 4),
        "unanswerable_failure_rate": (1, 4),
    }
    # Failures are never refusals, and each answerable item lands in exactly one bucket.
    rr = rv.rates(items)
    assert (
        rr["over_refusal_rate"].count
        + rr["answer_coverage"].count
        + rr["answerable_failure_rate"].count
        == 8
    )
    counts = rv.outcome_counts(items)
    assert counts["answerable"] == {
        "answered": 2, "partial": 1, "refused": 2, "invalid_output": 1, "timeout": 1, "crashed": 1
    }  # fmt: skip
    assert counts["unanswerable"]["refused"] == 2 and counts["unanswerable"]["crashed"] == 1


def test_empty_classes_are_not_calculable() -> None:
    items = rv.score([record(A, ANSWERED)], [A])
    r = rv.rates(items)
    assert r["correct_refusal_rate"].value is None
    assert str(r["correct_refusal_rate"]) == "n/a (0/0)"
    assert r["answer_coverage"].value == 1.0
    assert rv.rates([])["over_refusal_rate"].value is None
    # Only failures: the completed-only variant has nothing to divide by.
    only_crash = rv.score([record(A, error="RuntimeError: x")], [A])
    assert rv.rates(only_crash)["over_refusal_rate_completed"].value is None
    assert rv.rates(only_crash)["over_refusal_rate"].value == 0.0


def test_by_category_and_over_refusal_breakdown() -> None:
    d = gold("q003", category="definition")
    recs = [
        record(A, ANSWERED),
        record(d, refused()),
        record(U, refused("out_of_scope"), guard_label="other_law"),
    ]
    items = rv.score(recs, [A, d, U])
    cats = rv.by_category(items)
    assert cats["definition"]["gold"] == "answerable"
    assert (cats["definition"]["refused"].count, cats["definition"]["refused"].denominator) == (
        1,
        1,
    )
    assert cats["unanswerable"]["gold"] == "unanswerable"
    assert cats["lookup"]["substantive"].value == 1.0
    br = rv.over_refusal_breakdown(items)
    # The correct refusal of U is not an over-refusal.
    assert br["source"] == {"draft": 1} and br["reason"] == {"insufficient_evidence": 1}


# ------------------------------------------------------------------------ trusted labels


def test_labels_come_from_the_golden_set_and_mismatches_stop_scoring() -> None:
    rec = record(A, refused())  # refused, but gold says answerable: an over-refusal
    assert rv.score([rec], [A])[0].should_refuse is False
    relabelled = gold("q001", should_refuse=True)  # the golden set now says otherwise
    with pytest.raises(rv.LabelError, match="q001"):
        rv.score([rec], [relabelled])
    with pytest.raises(rv.LabelError, match="not in the golden set"):
        rv.score([rec], [U])
    with pytest.raises(rv.LabelError, match="2 records"):
        rv.score([rec, rec], [A])


# ------------------------------------------------------------------------ CLI


def _write_run(tmp: Path, recs: list[AnswerRecord], meta: dict[str, Any] | None) -> Path:
    run = tmp / "run"
    run.mkdir()
    (run / "answers.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in recs), encoding="utf-8"
    )
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return run


def _write_golden(tmp: Path, items: list[GoldenItem]) -> Path:
    path = tmp / "golden.jsonl"
    path.write_text("".join(g.model_dump_json() + "\n" for g in items), encoding="utf-8")
    return path


def test_cli_reports_rates_and_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from citeshield.evaluation.run_eval import golden_sha

    meta = {"run_id": "r1", "subset": "x", "ids": ["q001", "q002", "q009"],
            "golden_sha": golden_sha([A, U, gold("q009")])}  # fmt: skip
    run = _write_run(tmp_path, [record(A, refused()), record(U, refused("out_of_scope"))], meta)
    out = tmp_path / "report.md"
    golden = _write_golden(tmp_path, [A, U, gold("q009")])
    assert rv.main([str(run), "--golden", str(golden), "--out", str(out)]) == 0
    text = out.read_text("utf-8")
    assert "| over_refusal_rate | 1.000 (1/1) |" in text
    assert "| correct_refusal_rate | 1.000 (1/1) |" in text
    assert "not run: q009" in text and "(unchanged: True)" in text
    assert "Over-refused ids: q001." in capsys.readouterr().out


def test_cli_refuses_records_that_disagree_with_the_golden_set(tmp_path: Path) -> None:
    run = _write_run(tmp_path, [record(A, ANSWERED)], None)
    golden = _write_golden(tmp_path, [gold("q001", category="definition")])
    assert rv.main([str(run), "--golden", str(golden)]) == 2


def test_reviewed_only_scores_human_reviewed_items(tmp_path: Path) -> None:
    drafted = gold("q003").model_copy(update={"reviewed": False})
    run = _write_run(tmp_path, [record(A, refused()), record(drafted, ANSWERED)], None)
    all_items = rv.evaluate_run(run, [A, drafted])
    reviewed = rv.evaluate_run(run, [A, drafted], reviewed_only=True)
    assert all_items["rates"]["over_refusal_rate"].denominator == 2
    assert reviewed["rates"]["over_refusal_rate"].denominator == 1
    assert reviewed["scope"] == "human-reviewed golden items only (1 of 2)"


def test_over_refusals_after_a_rejected_first_draft() -> None:
    a2, a3 = gold("q003"), gold("q004")
    recs = [
        record(A, refused(), first_draft_valid=False),  # refused while repairing a draft
        record(a2, refused(), first_draft_valid=True),  # refused at the first draft
        record(a3, ANSWERED, first_draft_valid=False),  # repaired into an answer
        record(U, refused("out_of_scope"), guard_label="other_law"),
    ]
    br = rv.over_refusal_breakdown(rv.score(recs, [A, a2, a3, U]))
    assert br["source"] == {"draft": 2}
    assert br["after_rejected_draft"] == {"draft": 1}
