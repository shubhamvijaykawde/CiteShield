"""Answer evals: records, metrics, smoke selection, judge, CI gate. Labelling: test_labelling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from citeshield.config import PROJECT_ROOT
from citeshield.contracts.answer import AgentAnswer
from citeshield.evaluation import gate, metrics
from citeshield.evaluation.golden import Category, GoldenItem, load_golden
from citeshield.evaluation.judge import Judge, JudgeVerdict, judge_prompt
from citeshield.evaluation.records import AnswerRecord, CitedSource
from citeshield.evaluation.run_eval import golden_sha, select, stratified_smoke
from helpers import json_turn, make_chunk

C61 = make_chunk("GDPR", 6, "1. Processing shall be lawful only if ...", paragraph=1)
C62 = make_chunk("GDPR", 6, "2. Member States may maintain ...", paragraph=2)
C7 = make_chunk("GDPR", 7, "1. Where processing is based on consent ...", paragraph=1)
CHUNKS = {c.chunk_id: c for c in (C61, C62, C7)}
lookup = CHUNKS.get


def answered(*chunk_ids: str) -> AgentAnswer:
    return AgentAnswer.model_validate(
        {
            "status": "answered",
            "summary": "Processing needs a legal basis.",
            "claims": [
                {
                    "text": f"Claim {i}.",
                    "citations": [{"chunk_id": cid, "quote": "shall be lawful"}],
                }
                for i, cid in enumerate(chunk_ids)
            ],
        }
    )


REFUSED = AgentAnswer.model_validate(
    {"status": "refused", "summary": "Out of scope.", "refusal_reason": "out_of_scope"}
)


def record(
    rid: str = "q001",
    *,
    answer: AgentAnswer | None = None,
    expected: tuple[str, ...] = ("GDPR 6(1)", "GDPR 7(1)"),
    should_refuse: bool = False,
    error: str | None = None,
    **kw: Any,
) -> AnswerRecord:
    return AnswerRecord.model_validate(
        {
            "id": rid,
            "category": "unanswerable" if should_refuse else "lookup",
            "question": "Q?",
            "should_refuse": should_refuse,
            "expected": [] if should_refuse else list(expected),
            "reference_answer": None if should_refuse else "Ref.",
            "answer": answer,
            "error": error,
            "schema_valid": answer is not None,
            "first_draft_valid": answer is not None,
            **kw,
        }
    )


# ------------------------------------------------------------------------ metrics


def test_citation_and_context_metrics() -> None:
    rec = record(
        answer=answered(C61.chunk_id, C62.chunk_id),
        retrieved=[C62.chunk_id, C61.chunk_id, C7.chunk_id],
    )
    assert metrics.citation_precision(rec, lookup) == 0.5  # 6(2) is not expected
    assert metrics.citation_recall(rec, lookup) == 0.5  # 7(1) never cited
    # Relevant at ranks 2 and 3: mean(1/2, 2/3).
    assert metrics.context_precision(rec, lookup) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert metrics.context_recall(rec, lookup) == 1.0


def test_paragraph_refs_match_points() -> None:
    pt = make_chunk("GDPR", 6, "(a) the data subject has given consent", paragraph=1, point="a")
    rec = record(answer=answered(pt.chunk_id), expected=("GDPR 6(1)",))
    assert metrics.citation_precision(rec, {pt.chunk_id: pt}.get) == 1.0


def test_refusals_and_crashes() -> None:
    recs = [
        record("q1", answer=answered(C61.chunk_id)),
        record("q2", answer=REFUSED),  # wrong refusal
        record("q3", error="LLMContextExceeded: x", latency_ms=99_000),  # crash, not a refusal
        record("q4", answer=REFUSED, should_refuse=True),
        record("q5", answer=answered(C61.chunk_id), should_refuse=True),  # missed refusal
    ]
    assert recs[2].crashed and not recs[2].refused and not recs[2].judgeable
    m = metrics.compute(recs, lookup)
    # The crash is excluded: tp q4, fp q2, fn q5.
    assert (m["refusal_precision"], m["refusal_recall"], m["refusal_f1"]) == (0.5, 0.5, 0.5)
    assert m["schema_validity"] == 0.8  # the crash has no valid answer
    assert (m["error_rate"], m["n_crashed"], m["n_completed"]) == (0.2, 1.0, 4.0)
    assert m["citation_recall"] == pytest.approx(0.25)  # q1 half, q2 zero; q3 not scored
    cats = metrics.by_category(recs, lookup)
    assert cats["lookup"]["crashed"] == 1.0
    assert cats["lookup"]["refused"] == pytest.approx(1 / 2)  # q2 of q1, q2
    assert m["judge_n"] == 0.0 and m["judge_correctness"] is None


def test_latency_ignores_crashes_and_uses_nearest_rank() -> None:
    recs = [
        record(f"q{i}", answer=answered(C61.chunk_id), latency_ms=ms)
        for i, ms in enumerate((1000, 2000, 9000))
    ]
    recs.append(record("qx", error="E: x", latency_ms=500_000))
    m = metrics.compute(recs, lookup)
    assert (m["latency_p50_s"], m["latency_p95_s"]) == (2.0, 9.0)


def test_judge_metrics() -> None:
    recs = [record("q1", answer=answered(C61.chunk_id)), record("q2", answer=answered(C7.chunk_id))]
    v = {"model": "m", "prompt_version": "p", "view_sha": "x"}
    judged = {
        "q1": JudgeVerdict(id="q1", correctness=3, faithful=True, citation_adequate=False, **v),
        "q2": JudgeVerdict(id="q2", correctness=2, faithful=False, citation_adequate=False, **v),
    }
    m = metrics.compute(recs, lookup, judged)
    assert m["judge_n"] == 2 and m["judge_correctness"] == 2.5
    assert m["judge_faithful"] == 0.5 and m["judge_citation_adequate"] == 0.0


def test_render_and_view() -> None:
    rec = record(
        answer=answered(C61.chunk_id),
        cited={C61.chunk_id: CitedSource(label="GDPR Art. 6(1)", text=C61.text)},
    )
    text = rec.render_answer()
    assert '[GDPR Art. 6(1)] "shall be lawful"' in text
    view = rec.view()
    assert "<reference_answer>\nRef.\n" in view and C61.text in view
    assert rec.view_sha == record(answer=answered(C61.chunk_id), cited=rec.cited).view_sha
    assert rec.view_sha != record(answer=answered(C7.chunk_id), cited=rec.cited).view_sha
    assert rec.judgeable
    assert not record(answer=REFUSED).judgeable


def test_old_run_records_still_load() -> None:
    line = (
        '{"id":"q054","category":"unanswerable","question":"Tax?","should_refuse":true,'
        '"expected":[],"reference_answer":null,"answer":{"status":"refused","summary":"No.",'
        '"claims":[],"checklist":null,"refusal_reason":"out_of_scope","caveats":[],'
        '"corpus_version":"x"},"error":null,"guard_label":"other_law","cited":{},'
        '"retrieved":[],"support_ratio":null,"first_draft_valid":true,"schema_valid":true,'
        '"latency_ms":6310.8,"cost_usd":0.0,"input_tokens":288,"output_tokens":357,"llm_calls":1}'
    )
    assert AnswerRecord.model_validate_json(line).refused


# ------------------------------------------------------------------------ selection


@pytest.fixture(scope="module")
def golden() -> list[GoldenItem]:
    return load_golden(PROJECT_ROOT / "eval" / "golden" / "questions.jsonl")


def test_stratified_smoke(golden: list[GoldenItem]) -> None:
    ids = stratified_smoke(golden)
    assert len(ids) == 40 == len(set(ids))
    assert ids == stratified_smoke(list(reversed(golden)))  # order-independent
    by_id = {i.id: i for i in golden}
    picked = [by_id[i].category for i in ids]
    assert set(picked) == {i.category for i in golden}
    assert all(picked.count(c) >= 2 for c in set(picked))
    share = picked.count(Category.UNANSWERABLE) / len(ids)
    assert abs(share - sum(i.should_refuse for i in golden) / len(golden)) < 0.05


def test_committed_smoke_subset_is_valid(golden: list[GoldenItem]) -> None:
    items = select(golden, "smoke", None)
    assert len(items) == 40
    assert {i.category for i in items} == {i.category for i in golden}


def test_select_and_golden_sha(golden: list[GoldenItem]) -> None:
    two = select(golden, "full", "q002,q054")
    assert [i.id for i in two] == ["q002", "q054"]
    with pytest.raises(ValueError, match="q999"):
        select(golden, "full", "q999")
    assert golden_sha(two) == golden_sha(list(reversed(two)))
    edited = [two[0].model_copy(update={"question": "changed"}), two[1]]
    assert golden_sha(edited) != golden_sha(two)


# ------------------------------------------------------------------------ judge


class FakeJudgeLLM:
    def __init__(self, grades: dict[str, Any]) -> None:
        self.grades = grades
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return json_turn({"rationale": "ok", **self.grades})


def test_judge_grades_the_view() -> None:
    llm = FakeJudgeLLM({"correctness": 3, "faithful": True, "citation_adequate": True})
    rec = record(answer=answered(C61.chunk_id))
    v = Judge(llm, "judge-model").grade(rec)
    assert (v.correctness, v.faithful, v.view_sha) == (3, True, rec.view_sha)
    assert v.prompt_version == judge_prompt().version
    call = llm.calls[0]
    assert call["messages"][0]["content"] == rec.view()
    assert call["json_schema"]["properties"]["correctness"]["enum"] == [1, 2, 3]


# ------------------------------------------------------------------------ gate


def _metrics_doc(**over: float | None) -> dict[str, Any]:
    m: dict[str, float | None] = {
        "n": 40.0,
        "faithfulness": 0.95,
        "citation_precision": 0.90,
        "refusal_f1": 0.90,
        "schema_validity": 1.0,
        "citation_recall": 0.8,
    }
    m.update(over)
    meta = {
        "run_id": "r",
        "subset": "smoke",
        "golden_sha": "g",
        "corpus_hash": "c",
        "agent": {
            "llm_provider": "anthropic",
            "agent_model": "a",
            "guard_model": "g",
            "verifier_model": "v",
        },
        "runtime": {"models": {"agent_model": "a"}, "llm_options": {"num_ctx": 24576}},
        "num_ctx_observed": [24576],
        "config_consistent": True,
    }
    return {"meta": meta, "metrics": m}


@pytest.mark.parametrize(
    ("over", "status"),
    [
        ({}, "pass"),
        ({"faithfulness": 0.925, "citation_precision": 0.875, "refusal_f1": 0.855}, "pass"),
        ({"faithfulness": 0.91}, "fail"),  # drop 0.04
        ({"citation_precision": 0.86}, "fail"),
        ({"refusal_f1": 0.84}, "fail"),
        ({"schema_validity": 0.975}, "fail"),
        ({"faithfulness": None}, "fail"),
        ({"citation_recall": 0.1}, "pass"),  # reported, not gated
    ],
)
def test_gate_rules(over: dict[str, float | None], status: str) -> None:
    result = gate.evaluate(_metrics_doc(), _metrics_doc(**over))
    assert result.status == status
    assert result.exit_code == {"pass": 0, "fail": 1}[status]


def test_gate_absolute_faithfulness_floor() -> None:
    base = _metrics_doc(faithfulness=0.86)
    assert gate.evaluate(base, _metrics_doc(faithfulness=0.84)).status == "fail"


def test_gate_not_comparable() -> None:
    cur = _metrics_doc()
    cur["meta"] = {
        **cur["meta"],
        "subset": "full",
        "agent": {**cur["meta"]["agent"], "agent_model": "b"},
    }
    result = gate.evaluate(_metrics_doc(), cur)
    assert result.status == "not_comparable" and result.exit_code == 2
    assert len(result.mismatches) == 2
    assert "NOT COMPARABLE" in result.markdown


def test_gate_cli_baseline_roundtrip(tmp_path: Path) -> None:
    run = tmp_path / "metrics.json"
    run.write_text(json.dumps(_metrics_doc()), "utf-8")
    base = tmp_path / "baseline.json"
    assert gate.main([str(run), "--baseline", str(base)]) == 2  # no baseline yet
    assert gate.main([str(run), "--baseline", str(base), "--write-baseline"]) == 0
    comment = tmp_path / "comment.md"
    assert gate.main([str(run), "--baseline", str(base), "--markdown", str(comment)]) == 0
    assert "| faithfulness | 0.950 | 0.950 | +0.000 | ok |" in comment.read_text("utf-8")


def test_gate_llm_options_and_mixed_runs_are_not_comparable() -> None:
    cur = _metrics_doc()
    cur["meta"] = {
        **cur["meta"],
        "runtime": {**cur["meta"]["runtime"], "llm_options": {"num_ctx": 16384}},
    }
    assert gate.evaluate(_metrics_doc(), cur).status == "not_comparable"
    mixed = _metrics_doc()
    mixed["meta"] = {
        **mixed["meta"],
        "config_consistent": False,
        "num_ctx_observed": [16384, 24576],
    }
    assert gate.evaluate(_metrics_doc(), mixed).status == "not_comparable"


def test_write_baseline_refuses_crashes_and_missing_runtime(tmp_path: Path) -> None:
    base = tmp_path / "baseline.json"
    crashed = tmp_path / "crashed.json"
    crashed.write_text(json.dumps(_metrics_doc(n_crashed=17.0)), "utf-8")
    assert gate.main([str(crashed), "--baseline", str(base), "--write-baseline"]) == 1
    old = _metrics_doc()
    del old["meta"]["runtime"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(old), "utf-8")
    assert gate.main([str(legacy), "--baseline", str(base), "--write-baseline"]) == 1
    assert not base.exists()


def _ollama_doc(**over: float | None) -> dict[str, Any]:
    doc = _metrics_doc(**{"n_crashed": 0.0, **over})
    doc["meta"]["agent"] = {**doc["meta"]["agent"], "llm_provider": "ollama"}
    return doc


@pytest.mark.parametrize(
    ("over", "status", "advisory"),
    [
        ({}, "pass", 0),
        # Quality drops (and the Claude-only 0.85 floor) are advisory for Ollama.
        ({"faithfulness": 0.70, "citation_precision": 0.70, "refusal_f1": 0.60}, "pass", 3),
        ({"schema_validity": 0.95}, "fail", 0),
        ({"n_crashed": 1.0}, "fail", 0),
    ],
)
def test_ollama_gate_blocks_only_deterministic_checks(
    over: dict[str, float | None], status: str, advisory: int
) -> None:
    result = gate.evaluate(_ollama_doc(), _ollama_doc(**over))
    assert result.status == status
    assert len(result.advisories) == advisory
    if advisory:
        assert "Advisory (not blocking)" in result.markdown


def test_ollama_has_no_absolute_faithfulness_floor() -> None:
    result = gate.evaluate(_ollama_doc(faithfulness=0.86), _ollama_doc(faithfulness=0.84))
    assert result.status == "pass" and not result.advisories


def test_anthropic_gate_keeps_the_absolute_floor() -> None:
    assert gate.evaluate(_metrics_doc(), _metrics_doc(faithfulness=0.84)).status == "fail"
    assert gate.rules_for(_metrics_doc()["meta"]) is gate.RULES


def test_provisional_baseline_is_labelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "BASELINES_DIR", tmp_path)
    run = tmp_path / "metrics.json"
    run.write_text(json.dumps(_ollama_doc()), "utf-8")
    args = [str(run), "--write-baseline", "--provisional", "--note", "14 questions only."]
    assert gate.main(args) == 0
    written = json.loads((tmp_path / "ollama_provisional.json").read_text("utf-8"))
    assert written["baseline_info"]["status"] == "provisional"
    assert written["baseline_info"]["notes"] == ["14 questions only."]
    assert not (tmp_path / "ollama.json").exists()
    md = gate.evaluate(written, _ollama_doc()).markdown
    assert "**Provisional baseline.** 14 questions only." in md


def test_local_subset_file(golden: list[GoldenItem]) -> None:
    items = select(golden, "local", None)
    cats = {i.category for i in items}
    assert {
        Category.LOOKUP,
        Category.MULTI_PARAGRAPH,
        Category.CROSS_REGULATION,
        Category.TEMPORAL,
        Category.CHECKLIST,
        Category.UNANSWERABLE,
    } <= cats
    assert any(i.should_refuse for i in items) and any(not i.should_refuse for i in items)
    doc = json.loads((PROJECT_ROOT / "eval" / "golden" / "smoke_local.json").read_text("utf-8"))
    assert set(doc["items"]) == set(doc["ids"]) == {i.id for i in items}
