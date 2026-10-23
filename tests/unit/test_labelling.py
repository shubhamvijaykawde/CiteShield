"""Labelling sample, annotation validation, the labelling CLI and judge calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from citeshield.agent.llm import LLMResponse, Usage
from citeshield.config import PROJECT_ROOT, Settings
from citeshield.contracts.answer import AgentAnswer
from citeshield.evaluation import judge_calibration as jc
from citeshield.evaluation.annotations import HumanLabel, LabelItem, load_labels, load_sample
from citeshield.evaluation.judge import Judge, JudgeVerdict, RefusalVerdict
from citeshield.evaluation.label_sample import (
    Candidate,
    Quotas,
    assign_splits,
    build_sample,
    collect,
    perturb_contradiction,
    perturb_unsupported,
    perturb_wrong_citation,
)
from citeshield.evaluation.labeling import coverage, label_sample, labelling_order
from citeshield.evaluation.records import AnswerRecord, CitedSource
from helpers import json_turn, make_chunk

C33 = make_chunk(
    "GDPR", 33, "1. The controller shall notify the breach within 72 hours.", paragraph=1
)
C34 = make_chunk(
    "GDPR", 34, "1. The controller shall tell the data subjects without delay.", paragraph=1
)
C6 = make_chunk("GDPR", 6, "1. Processing shall be lawful only if one basis applies.", paragraph=1)
C99 = make_chunk("GDPR", 99, "2. It shall apply from 25 May 2018.", paragraph=2)
CHUNKS = {c.chunk_id: c for c in (C33, C34, C6, C99)}
lookup = CHUNKS.get


def answer_rec(
    qid: str,
    claim: str,
    chunk: Any = C33,
    *,
    status: str = "answered",
    expected: tuple[str, ...] = ("GDPR 33(1)",),
) -> AnswerRecord:
    quote = chunk.text[3:40]
    answer = {
        "status": status,
        "summary": claim,
        "claims": [{"text": claim, "citations": [{"chunk_id": chunk.chunk_id, "quote": quote}]}],
    }
    if status == "partial":
        answer["refusal_reason"] = "insufficient_evidence"
    return AnswerRecord.model_validate(
        {
            "id": qid,
            "category": "lookup",
            "question": f"Question {qid}?",
            "should_refuse": False,
            "expected": list(expected),
            "reference_answer": "Within 72 hours.",
            "answer": answer,
            "cited": {chunk.chunk_id: {"label": chunk.citation_label, "text": chunk.text}},
            "retrieved": [chunk.chunk_id, C34.chunk_id, C6.chunk_id],
            "schema_valid": True,
            "first_draft_valid": True,
        }
    )


def refusal_rec(qid: str, *, should_refuse: bool) -> AnswerRecord:
    return AnswerRecord.model_validate(
        {
            "id": qid,
            "category": "unanswerable" if should_refuse else "lookup",
            "question": f"Question {qid}?",
            "should_refuse": should_refuse,
            "expected": [] if should_refuse else ["GDPR 33(1)"],
            "reference_answer": None if should_refuse else "Within 72 hours.",
            "answer": {
                "status": "refused",
                "summary": "I cannot answer this.",
                "refusal_reason": "out_of_scope",
            },
            "schema_valid": True,
            "first_draft_valid": True,
        }
    )


def candidates() -> list[Candidate]:
    answers = [answer_rec(f"q{n:03d}", f"Notify within 72 hours (case {n}).") for n in range(1, 9)]
    answers.append(answer_rec("q001", "Notify the authority within 72 hours.", status="partial"))
    answers.append(answer_rec("q002", "The controller must notify the authority."))
    refusals = [refusal_rec(f"q1{n:02d}", should_refuse=True) for n in range(3)]
    refusals += [refusal_rec(f"q2{n:02d}", should_refuse=False) for n in range(3)]
    return [Candidate("run-a", r) for r in answers + refusals]


QUOTAS = Quotas(real=6, per_perturbation=1, correct_refusals=2, wrong_refusals=2)


# ------------------------------------------------------------------------ sample


def test_sample_composition_and_split() -> None:
    items, shortfalls = build_sample(candidates(), lookup, QUOTAS)
    assert shortfalls == []
    by_outcome: dict[str, int] = {}
    for i in items:
        by_outcome[i.outcome] = by_outcome.get(i.outcome, 0) + 1
    assert by_outcome.pop("partial") == 1  # partial answers are always included
    assert by_outcome.pop("answered") == 5
    assert by_outcome == {
        "correct_refusal": 2,
        "wrong_refusal": 2,
        "perturbed_unsupported": 1,
        "perturbed_wrong_citation": 1,
        "perturbed_contradiction": 1,
    }
    synthetic = [i for i in items if i.origin == "synthetic"]
    assert all(i.perturbation for i in synthetic)
    # Every item of a question is on the same side of the split.
    sides: dict[str, set[str]] = {}
    for i in items:
        sides.setdefault(i.question_id, set()).add(i.split)
    assert all(len(s) == 1 for s in sides.values())
    assert {i.split for i in items} == {"dev", "holdout"}
    assert len({i.item_id for i in items}) == len(items)


def test_sample_is_deterministic() -> None:
    a, _ = build_sample(candidates(), lookup, QUOTAS)
    b, _ = build_sample(list(reversed(candidates())), lookup, QUOTAS)
    assert [i.model_dump_json() for i in a] == [i.model_dump_json() for i in b]


def test_shortfalls_are_reported() -> None:
    _, shortfalls = build_sample(candidates()[:3], lookup, Quotas(real=10, per_perturbation=1))
    assert any(s.startswith("real answers: 3 of 10") for s in shortfalls)
    assert any(s.startswith("correct refusals: 0 of") for s in shortfalls)


def test_collect_attaches_current_golden_text(tmp_path: Path) -> None:
    from citeshield.evaluation.golden import Category, GoldenItem

    run = tmp_path / "r1"
    run.mkdir()
    old = answer_rec("q001", "Notify within 72 hours.").model_copy(
        update={"reference_answer": None}
    )
    crashed = AnswerRecord.model_validate(
        {**old.model_dump(), "id": "q002", "answer": None, "error": "E: x"}
    )
    (run / "answers.jsonl").write_text(
        old.model_dump_json() + "\n" + crashed.model_dump_json() + "\n"
    )
    golden = {
        q: GoldenItem(
            id=q,
            question="Q?",
            category=Category.LOOKUP,
            expected=["GDPR 33(1)"],
            reference_answer="Within 72 hours.",
        )
        for q in ("q001", "q002")
    }
    got = collect(tmp_path, golden)
    assert [c.record.id for c in got] == ["q001"]  # crashes are not labellable
    assert got[0].record.reference_answer == "Within 72 hours."


def test_hidden_fields_are_not_in_the_view() -> None:
    items, _ = build_sample(candidates(), lookup, QUOTAS)
    for i in items:
        view = i.record.view()
        # The agent's own "Status: answered" is part of the answer; the item's provenance
        # labels are not.
        assert "perturbed_" not in view and "_refusal" not in view and "synthetic" not in view
        assert i.source_run not in view and i.split not in view.split("<answer>")[0]
        if i.perturbation:
            assert i.perturbation not in view


# ------------------------------------------------------------------------ perturbations


def test_contradiction_changes_quantities_not_references() -> None:
    rec = answer_rec("q001", "Under Article 33 the controller must notify within 72 hours.")
    new, what = perturb_contradiction(rec) or (None, "")
    assert new is not None and new.answer is not None
    assert "144 hours" in new.answer.claims[0].text and "Article 33" in new.answer.claims[0].text
    assert "144 hours" in new.answer.summary
    assert "'72 hours' -> '144 hours'" in what


def test_contradiction_shifts_dates_but_not_act_numbers() -> None:
    rec = answer_rec("q001", "Regulation (EU) 2016/679 applies from 25 May 2018.", C99)
    new, _ = perturb_contradiction(rec) or (None, "")
    assert new is not None and new.answer is not None
    assert new.answer.claims[0].text == "Regulation (EU) 2016/679 applies from 25 May 2019."


def test_contradiction_negates_a_modal_when_there_is_no_number() -> None:
    rec = answer_rec("q001", "The controller must notify the authority.")
    new, what = perturb_contradiction(rec) or (None, "")
    assert new is not None and new.answer is not None
    assert new.answer.claims[0].text == "The controller must not notify the authority."
    assert "negated" in what


def test_unsupported_and_wrong_citation() -> None:
    rec = answer_rec("q001", "Notify within 72 hours.")
    new, _ = perturb_unsupported(rec, 2) or (None, "")
    assert new is not None and new.answer is not None
    assert new.answer.claims[0].text.endswith("criminal offence under this Regulation.")
    swapped, what = perturb_wrong_citation(rec, lookup) or (None, "")
    assert swapped is not None and swapped.answer is not None
    cid = swapped.answer.claims[0].citations[0].chunk_id
    assert cid in (C34.chunk_id, C6.chunk_id)  # retrieved, not expected, not cited before
    assert set(swapped.cited) == {cid}  # the old source is gone, the new one is shown
    assert swapped.view_sha != rec.view_sha
    assert "non-expected" in what


def test_assign_splits_is_question_level_and_balanced() -> None:
    pairs = [(f"q{n}", "answered") for n in range(10)] + [("q0", "perturbed_unsupported")]
    split = assign_splits(pairs)  # type: ignore[arg-type]
    assert set(split) == {f"q{n}" for n in range(10)}
    assert abs(sum(1 for s in split.values() if s == "dev") - 5) <= 1


# ------------------------------------------------------------------------ annotations


def _item(kind: str = "answer") -> LabelItem:
    rec = (
        answer_rec("q001", "Notify within 72 hours.")
        if kind == "answer"
        else refusal_rec("q100", should_refuse=True)
    )
    outcome = "answered" if kind == "answer" else "correct_refusal"
    return LabelItem(
        item_id=f"{rec.id}-{rec.view_sha}",
        question_id=rec.id,
        kind=kind,
        origin="real",
        outcome=outcome,
        split="dev",
        source_run="r",
        record=rec,
        view_sha=rec.view_sha,
    )  # type: ignore[arg-type]


def test_label_item_validation() -> None:
    item = _item()
    with pytest.raises(ValueError, match="view_sha"):
        LabelItem.model_validate({**item.model_dump(), "view_sha": "stale"})
    with pytest.raises(ValueError, match="disagree"):
        LabelItem.model_validate({**item.model_dump(), "origin": "synthetic"})
    with pytest.raises(ValueError, match="kind"):
        LabelItem.model_validate({**item.model_dump(), "kind": "refusal"})


def test_human_label_grades_must_match_kind() -> None:
    ok = HumanLabel(
        item_id="a",
        view_sha="v",
        kind="answer",
        correctness=2,
        faithful=True,
        citation_adequate=False,
    )
    assert ok.grades() == {"correctness": 2, "faithful": True, "citation_adequate": False}
    with pytest.raises(ValueError, match="missing"):
        HumanLabel(item_id="a", view_sha="v", kind="answer", correctness=2)
    with pytest.raises(ValueError, match="not allowed"):
        HumanLabel(
            item_id="a", view_sha="v", kind="refusal", refusal_appropriate=True, correctness=3
        )
    with pytest.raises(ValueError):
        HumanLabel(
            item_id="a",
            view_sha="v",
            kind="answer",
            correctness=4,
            faithful=True,
            citation_adequate=True,
        )


def test_load_labels_reports_problems_and_keeps_the_latest(tmp_path: Path) -> None:
    a, r = _item(), _item("refusal")
    good = HumanLabel(
        item_id=a.item_id,
        view_sha=a.view_sha,
        kind="answer",
        correctness=1,
        faithful=False,
        citation_adequate=False,
    )
    newer = good.model_copy(update={"correctness": 3})
    lines = [
        good.model_dump_json(),
        "{not json",
        HumanLabel(
            item_id="nope", view_sha="x", kind="refusal", refusal_appropriate=True
        ).model_dump_json(),
        good.model_copy(update={"view_sha": "stale"}).model_dump_json(),
        HumanLabel(
            item_id=r.item_id,
            view_sha=r.view_sha,
            kind="answer",
            correctness=3,
            faithful=True,
            citation_adequate=True,
        ).model_dump_json(),
        newer.model_dump_json(),
    ]
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(lines) + "\n", "utf-8")
    labels, problems = load_labels(path, [a, r])
    assert labels == {a.item_id: newer}
    assert [p.split(":")[0] for p in problems] == ["line 2", "line 3", "line 4", "line 5"]
    assert (
        "unknown item" in problems[1] and "stale" in problems[2] and "refusal item" in problems[3]
    )


# ------------------------------------------------------------------------ labelling CLI


def _write_sample(tmp_path: Path, items: list[LabelItem]) -> Path:
    path = tmp_path / "sample.jsonl"
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), "utf-8")
    return path


def test_labelling_cli_both_kinds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    items = [_item(), _item("refusal")]
    sample, labels_path = _write_sample(tmp_path, items), tmp_path / "labels.jsonl"
    order = [i.kind for i in labelling_order(items)]
    script = {"answer": ["7", "2", "y", "n", "dropped qualifier"], "refusal": ["y", ""]}
    answers = iter([x for kind in order for x in script[kind]])
    assert label_sample(sample, labels_path, ask=lambda _p: next(answers), annotator="me") == 2
    labels, problems = load_labels(labels_path, items)
    assert not problems
    a = labels[items[0].item_id]
    assert (a.correctness, a.faithful, a.citation_adequate, a.note, a.annotator) == (
        2,
        True,
        False,
        "dropped qualifier",
        "me",
    )
    assert labels[items[1].item_id].refusal_appropriate is True
    out = capsys.readouterr().out
    assert "correct_refusal" not in out and "perturbed_" not in out and "origin" not in out
    # Nothing left to ask.
    assert label_sample(sample, labels_path, ask=lambda _p: "q") == 0
    assert "| all | all | 2 | 2 |" in coverage(items, labels)


def test_labelling_cli_skip_quit_and_split(tmp_path: Path) -> None:
    items = [_item(), _item("refusal")]
    sample, labels_path = _write_sample(tmp_path, items), tmp_path / "labels.jsonl"
    assert label_sample(sample, labels_path, ask=lambda _p: "s") == 0
    assert label_sample(sample, labels_path, split="holdout", ask=lambda _p: "q") == 0
    assert not labels_path.exists() or not labels_path.read_text().strip()


# ------------------------------------------------------------------------ calibration


def test_cohen_kappa_known_values() -> None:
    a = [True, True, False, False]
    assert jc.cohen_kappa(a, a, [False, True]) == 1.0
    assert jc.cohen_kappa(a, [True, False, True, False], [False, True]) == pytest.approx(0.0)
    h = [True] * 25 + [False] * 25  # textbook: 20/5/10/15 -> 0.4
    j = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert jc.cohen_kappa(h, j, [False, True]) == pytest.approx(0.4)
    near = jc.cohen_kappa([1, 2, 3, 3], [1, 2, 2, 3], [1, 2, 3], quadratic=True)
    far = jc.cohen_kappa([1, 2, 3, 3], [1, 2, 1, 3], [1, 2, 3], quadratic=True)
    assert near is not None and far is not None and near > far
    assert jc.cohen_kappa([], [], [1, 2, 3]) is None
    assert jc.cohen_kappa([True, True], [True, True], [False, True]) is None  # undefined


def test_bootstrap_ci_is_reproducible_and_brackets_kappa() -> None:
    h = [True] * 25 + [False] * 25
    j = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    ci = jc.bootstrap_ci(h, j, [False, True], n_boot=500)
    assert ci == jc.bootstrap_ci(h, j, [False, True], n_boot=500)
    assert ci is not None and ci[0] < 0.4 < ci[1]
    assert jc.bootstrap_ci([True], [True], [False, True]) is None


def _pairs(n: int, *, agree: bool = True, split: str = "holdout") -> list[jc.Pair]:
    out = []
    base = _item()
    for k in range(n):
        item = base.model_copy(update={"item_id": f"x{k}", "split": split})
        human = k % 2 == 0
        label = HumanLabel(
            item_id=item.item_id,
            view_sha=item.view_sha,
            kind="answer",
            correctness=1 + k % 3,
            faithful=human,
            citation_adequate=human,
        )
        judged = human if agree or k % 4 else not human
        verdict = JudgeVerdict(
            id="q001",
            view_sha=item.view_sha,
            correctness=1 + k % 3,
            faithful=judged,
            citation_adequate=judged,
            model="m",
            prompt_version="p",
        )
        out.append(jc.Pair(item, label, verdict))
    return out


def test_verdicts_are_conservative() -> None:
    perfect = jc.calibrate(_pairs(30))
    assert perfect["verdicts"]["faithful"] == "meets target"
    assert perfect["verdicts"]["refusal_appropriate"] == "insufficient data"  # no refusal pairs
    small = jc.calibrate(_pairs(10))
    assert small["verdicts"]["faithful"] == "insufficient data"
    noisy = jc.calibrate(_pairs(40, agree=False))
    assert noisy["verdicts"]["faithful"] == "below target"


def test_calibration_report_hides_holdout_disagreements() -> None:
    pairs = _pairs(24, agree=False) + _pairs(8, agree=False, split="dev")
    result = jc.calibrate(pairs)
    assert result["holdout_disagreements"] > 0
    assert all(d["item_id"].startswith("x") for d in result["dev_disagreements"])
    md = jc.markdown(
        result,
        {
            "model": "m",
            "prompt_versions": {"answer": "a", "refusal": "r"},
            "n_labels": 32,
            "n_items": 60,
        },
        [],
    )
    assert "Held-out disagreements:" in md and "not listed" in md
    assert "| faithful | holdout | 24 |" in md


class FakeJudgeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        props = kwargs["json_schema"]["properties"]
        if "refusal_appropriate" in props:
            return json_turn({"rationale": "scope", "refusal_appropriate": True})
        return json_turn(
            {"rationale": "ok", "correctness": 3, "faithful": True, "citation_adequate": True}
        )


def test_pair_up_judges_once_per_item_and_caches(tmp_path: Path) -> None:
    items = [_item(), _item("refusal")]
    labels = {
        items[0].item_id: HumanLabel(
            item_id=items[0].item_id,
            view_sha=items[0].view_sha,
            kind="answer",
            correctness=3,
            faithful=True,
            citation_adequate=True,
        ),
        items[1].item_id: HumanLabel(
            item_id=items[1].item_id,
            view_sha=items[1].view_sha,
            kind="refusal",
            refusal_appropriate=True,
        ),
    }
    llm, verdicts = FakeJudgeLLM(), tmp_path / "verdicts.jsonl"
    judge = Judge(llm, "judge-model")
    pairs, skipped = jc.pair_up(items, labels, judge, verdicts, "judge-model")
    assert len(pairs) == 2 and not skipped and len(llm.calls) == 2
    assert isinstance(pairs[1].verdict, RefusalVerdict)
    again, _ = jc.pair_up(items, labels, judge, verdicts, "judge-model")
    assert len(again) == 2 and len(llm.calls) == 2  # cached
    none, skipped = jc.pair_up(items, labels, None, verdicts, "other-model")
    assert not none and len(skipped) == 2


def test_judge_rejects_truncated_output() -> None:
    class Truncated:
        def chat(self, **kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content=[],
                text="",
                tool_calls=[],
                stop_reason="max_tokens",
                model="m",
                usage=Usage(),
            )

    with pytest.raises(ValueError, match="budget"):
        Judge(Truncated(), "m", max_tokens=10).grade(answer_rec("q001", "x y z"))


def test_ollama_judge_gets_the_full_output_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from citeshield.evaluation.run_eval import build_judge

    monkeypatch.setenv("CITESHIELD_LLM__PROVIDER", "ollama")
    judge = build_judge(Settings(), None)
    assert judge.max_tokens == 8192 and judge.model == "qwen3:4b"


# ------------------------------------------------------------------------ committed files


def test_committed_sample_is_valid() -> None:
    items = load_sample(PROJECT_ROOT / "eval" / "labels" / "sample.jsonl")
    assert len(items) == 60
    assert sum(i.origin == "synthetic" for i in items) == 12
    assert sum(i.kind == "refusal" for i in items) == 12
    sides: dict[str, set[str]] = {}
    for i in items:
        sides.setdefault(i.question_id, set()).add(i.split)
    assert all(len(s) == 1 for s in sides.values())
    assert sum(i.split == "dev" for i in items) == 30
    assert (PROJECT_ROOT / "eval" / "labels" / "GUIDE.md").exists()


def test_refusal_answer_contract_round_trips() -> None:
    rec = refusal_rec("q100", should_refuse=True)
    assert rec.refused and not rec.crashed
    assert AgentAnswer.model_validate_json(rec.answer.model_dump_json()) == rec.answer  # type: ignore[union-attr]
    assert json.loads(rec.model_dump_json())["answer"]["status"] == "refused"
    assert CitedSource(label="x", text="y").label == "x"


def test_relabel_one_item_replaces_only_its_label(tmp_path: Path) -> None:
    items = [_item(), _item("refusal")]
    sample, labels_path = _write_sample(tmp_path, items), tmp_path / "labels.jsonl"
    order = [i.kind for i in labelling_order(items)]
    script = {"answer": ["3", "y", "y", "first"], "refusal": ["y", ""]}
    answers = iter([x for kind in order for x in script[kind]])
    label_sample(sample, labels_path, ask=lambda _p: next(answers))
    before, _ = load_labels(labels_path, items)
    fix = iter(["1", "n", "n", "corrected"])
    assert label_sample(sample, labels_path, item=items[0].item_id, ask=lambda _p: next(fix)) == 1
    after, problems = load_labels(labels_path, items)
    assert not problems
    assert after[items[0].item_id].grades() == {
        "correctness": 1,
        "faithful": False,
        "citation_adequate": False,
    }
    assert after[items[1].item_id] == before[items[1].item_id]
    assert len(labels_path.read_text().splitlines()) == 3  # appended, history kept
    with pytest.raises(SystemExit, match="unknown item"):
        label_sample(sample, labels_path, item="nope", ask=lambda _p: "q")


def test_calibration_report_survives_a_cp1252_console(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import sys

    item = _item()
    sample = _write_sample(tmp_path, [item])
    label = HumanLabel(
        item_id=item.item_id,
        view_sha=item.view_sha,
        kind="answer",
        correctness=2,
        faithful=True,
        citation_adequate=True,
        note="kappa \u2265 0.7, \u201cquoted\u201d",
    )
    (tmp_path / "labels.jsonl").write_text(label.model_dump_json() + "\n", "utf-8")
    verdict = JudgeVerdict(
        id=item.record.id,
        view_sha=item.view_sha,
        correctness=3,
        faithful=True,
        citation_adequate=True,
        rationale="\u2265 all points",
        model="m",
        prompt_version=jc.prompt_versions()["answer"],
    )
    (tmp_path / "verdicts.jsonl").write_text(verdict.model_dump_json() + "\n", "utf-8")
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    args = [
        "--cached",
        "--sample",
        str(sample),
        "--labels",
        str(tmp_path / "labels.jsonl"),
        "--verdicts",
        str(tmp_path / "verdicts.jsonl"),
        "--out-dir",
        str(tmp_path),
    ]
    assert jc.main(args) == 0
    assert "\u2265" in (tmp_path / "judge_calibration.md").read_text("utf-8")


def test_frozen_judge_prompts_are_unchanged() -> None:
    import hashlib

    frozen = json.loads((PROJECT_ROOT / "eval" / "labels" / "frozen_judge.json").read_text())
    for spec in frozen["prompts"].values():
        digest = hashlib.sha256((PROJECT_ROOT / spec["file"]).read_bytes()).hexdigest()
        assert digest == spec["sha256"], f"{spec['file']} changed after the freeze"
        assert digest.startswith(spec["version"])
    assert jc.frozen_mismatches(frozen, frozen["model"]) == []
    assert jc.frozen_mismatches(frozen, "other-model") == [
        "judge model is other-model, frozen qwen3:4b"
    ]


def test_calibration_refuses_an_unfrozen_judge(tmp_path: Path) -> None:
    item = _item()
    sample = _write_sample(tmp_path, [item])
    label = HumanLabel(
        item_id=item.item_id,
        view_sha=item.view_sha,
        kind="answer",
        correctness=2,
        faithful=True,
        citation_adequate=True,
    )
    (tmp_path / "labels.jsonl").write_text(label.model_dump_json() + "\n", "utf-8")
    frozen = {
        "model": "qwen3:4b",
        "prompts": {"answer": {"version": "00000000"}, "refusal": {"version": "11111111"}},
    }
    (tmp_path / "frozen.json").write_text(json.dumps(frozen), "utf-8")
    args = [
        "--cached",
        "--judge-model",
        "qwen3:4b",
        "--sample",
        str(sample),
        "--labels",
        str(tmp_path / "labels.jsonl"),
        "--verdicts",
        str(tmp_path / "v.jsonl"),
        "--out-dir",
        str(tmp_path),
        "--frozen",
        str(tmp_path / "frozen.json"),
    ]
    assert jc.main(args) == 2
    assert not (tmp_path / "judge_calibration.md").exists()
    assert jc.main([*args, "--allow-unfrozen"]) == 0
    assert "NOT the frozen judge" in (tmp_path / "judge_calibration.md").read_text("utf-8")
