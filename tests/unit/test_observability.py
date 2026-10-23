"""Optional observability: Langfuse tracing and MLflow logging, off by default.

Fakes stand in for the SDKs (the default install has neither); one test uses the real
MLflow when the 'observability' extra is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

from citeshield import observability as obs
from citeshield.agent.config import AgentConfig
from citeshield.agent.graph import CiteShieldAgent
from citeshield.agent.tools import ToolExecutor
from citeshield.config import IngestConfig
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.validate import validate_document
from citeshield.retrieval.config import RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.expand import Expander
from citeshield.retrieval.hybrid import HybridSearcher
from citeshield.retrieval.index import build_index
from citeshield.retrieval.pipeline import Retriever
from helpers import HashingEncoder, ScriptedLLM, entail_all, json_turn, text_turn, tool_turn

V = "@oj-2099-01-01"
P2 = f"32099R0001:art2:p2{V}"
QUOTE = "shall not apply to widgets used solely for research"
CFG = AgentConfig(agent_model="agent-model", guard_model="guard-model", max_tool_rounds=3)
DRAFT = {
    "status": "answered",
    "summary": "Research widgets are exempt.",
    "claims": [{"text": "Research widgets are exempt.",
                "citations": [{"chunk_id": P2, "quote": QUOTE}]}],
    "checklist": [],
    "refusal_reason": None,
    "caveats": [],
}  # fmt: skip


@pytest.fixture
def executor(parsed: ParseResult) -> ToolExecutor:
    corpus = Corpus(validate_document(parsed.records, parsed.toc, IngestConfig()).valid)
    cfg = RetrieverConfig(collection="test")
    client = QdrantClient(":memory:")
    build_index(client, corpus, HashingEncoder(), cfg)
    searcher = HybridSearcher(client, corpus, HashingEncoder(), cfg)
    return ToolExecutor(Retriever(searcher, None, Expander(corpus), cfg))


class FakeObservation:
    def __init__(self, log: list[dict[str, Any]], **kw: Any) -> None:
        self.log, self.kw, self.ended = log, kw, False
        self.updates: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        log.append(self.__dict__)

    def start_observation(self, **kw: Any) -> FakeObservation:
        return FakeObservation(self.log, **kw)

    def update(self, **kw: Any) -> FakeObservation:
        self.updates.append(kw)
        return self

    def create_event(self, **kw: Any) -> None:
        self.events.append(kw)

    def end(self) -> FakeObservation:
        self.ended = True
        return self


class FakeLangfuse:
    def __init__(self) -> None:
        self.log: list[dict[str, Any]] = []
        self.flushed = False

    def start_observation(self, **kw: Any) -> FakeObservation:
        return FakeObservation(self.log, **kw)

    def flush(self) -> None:
        self.flushed = True


def test_langfuse_needs_both_keys_and_can_be_switched_off() -> None:
    keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
    assert obs.langfuse_configured(keys)
    assert not obs.langfuse_configured({"LANGFUSE_PUBLIC_KEY": "pk"})
    assert not obs.langfuse_configured({})
    assert not obs.langfuse_configured({**keys, "CITESHIELD_TRACING": "off"})


def test_without_langfuse_the_emit_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

    def emit(kind: str, payload: dict[str, Any]) -> None: ...

    with obs.traced(emit, "q", {}, {}) as e:
        assert e is emit


def _traced_run(executor: ToolExecutor, client: FakeLangfuse, llm: ScriptedLLM) -> list[str]:
    agent = CiteShieldAgent(llm, executor, CFG, verifier=entail_all(executor))
    seen: list[str] = []
    inputs = {"regulations": None, "mode": "answer"}
    with obs.traced(
        lambda k, p: seen.append(k), "Are widgets exempt?", inputs, agent.meta, client
    ) as emit:
        agent.run("Are widgets exempt?", emit=emit)
    return seen


def test_trace_covers_guard_tools_llm_calls_and_outcome(executor: ToolExecutor) -> None:
    client = FakeLangfuse()
    llm = ScriptedLLM(
        agent_turns=[
            tool_turn(("retrieve", {"query": "research widgets", "regulations": []})),
            tool_turn(("get_article", {"regulation": "Test Reg", "article": 2, "paragraph": 2})),
            text_turn("ok"),
            json_turn(DRAFT),
        ]
    )
    seen = _traced_run(executor, client, llm)
    assert seen[-1] == "final"  # the caller still gets every event
    by_type = {o["kw"].get("as_type"): o for o in client.log}
    root = by_type["agent"]
    assert root["kw"]["input"]["question"] == "Are widgets exempt?"
    assert root["ended"] and client.flushed
    assert root["updates"][-1]["output"]["status"] == "answered"
    assert root["updates"][-1]["output"]["cited"] == ["Test Reg Art. 2(2)"]
    assert by_type["guardrail"]["updates"][0]["output"] == {"label": "in_scope"}
    retriever = by_type["retriever"]
    assert retriever["kw"]["input"]["query"] == "research widgets" and retriever["ended"]
    gens = [o for o in client.log if o["kw"].get("as_type") == "generation"]
    assert [g["kw"]["name"] for g in gens] == ["llm.guard"] + ["llm.agent"] * 3 + ["llm.draft"]
    # The scripted provider reports no token counts, so none are invented.
    assert all(g["kw"]["usage_details"] is None for g in gens)


def test_trace_marks_failures_without_breaking_the_request(executor: ToolExecutor) -> None:
    client = FakeLangfuse()
    with pytest.raises(AssertionError, match="ran out of agent turns"):
        _traced_run(executor, client, ScriptedLLM())
    root = next(o for o in client.log if o["kw"].get("as_type") == "agent")
    assert root["updates"][-1]["level"] == "ERROR" and root["ended"]


def test_tracing_errors_are_swallowed(executor: ToolExecutor) -> None:
    class Broken(FakeLangfuse):
        def start_observation(self, **kw: Any) -> FakeObservation:
            ob = super().start_observation(**kw)

            def fail(**_: Any) -> FakeObservation:
                raise RuntimeError("exporter down")

            ob.start_observation = fail  # type: ignore[method-assign]
            return ob

    read_p2 = ("get_article", {"regulation": "Test Reg", "article": 2, "paragraph": 2})
    turns = [tool_turn(("retrieve", {"query": "w", "regulations": []})), tool_turn(read_p2)]
    llm = ScriptedLLM(agent_turns=[*turns, text_turn("ok"), json_turn(DRAFT)])
    assert _traced_run(executor, Broken(), llm)[-1] == "final"


# ------------------------------------------------------------------------ MLflow

META = {
    "run_id": "r1",
    "subset": "local",
    "golden_sha": "g",
    "agent": {"llm_provider": "ollama", "agent_model": "qwen3:4b",
              "prompt_versions": {"draft": "a1a7ec56"},
              "verifier_thresholds": {"entail": 0.51}},
    "runtime": {"llm_options": {"num_ctx": 24576}, "code": {"git_commit": "abc"},
                "models": {"embed_model": "BAAI/bge-m3", "ollama_digests": {"qwen3:4b": "d"}}},
}  # fmt: skip
METRICS = {"faithfulness": 0.889, "cost_per_question_usd": 0.0, "judge_correctness": None,
           "n": 14.0, "flag": True}  # fmt: skip


def test_params_and_metrics() -> None:
    params = obs.run_params(META)
    assert params["prompt.draft"] == "a1a7ec56" and params["verifier.entail"] == "0.51"
    assert params["llm_options.num_ctx"] == "24576" and params["code.git_commit"] == "abc"
    assert params["models.embed_model"] == "BAAI/bge-m3"
    m = obs.numeric_metrics(METRICS, "ollama")
    # None and bools are dropped; a computed (not billed) cost is not logged for Ollama.
    assert m == {"faithfulness": 0.889, "n": 14.0}
    assert "cost_per_question_usd" in obs.numeric_metrics(METRICS, "anthropic")


def test_log_eval_run_uses_local_store_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    calls: list[tuple[str, Any]] = []

    class Run:
        info = type("Info", (), {"run_id": "abc123"})()

        def __enter__(self) -> Run:
            return self

        def __exit__(self, *_: Any) -> None: ...

    class FakeMlflow:
        def __getattr__(self, name: str) -> Any:
            def call(*a: Any, **k: Any) -> Any:
                calls.append((name, a or k))
                return Run() if name == "start_run" else None

            return call

    run_id = obs.log_eval_run(META, METRICS, kind="answer-eval", mlflow=FakeMlflow(),
                              artifacts={"by_category.json": {"lookup": {"n": 3}}})  # fmt: skip
    assert run_id == "abc123"
    names = [n for n, _ in calls]
    assert names[:3] == ["set_tracking_uri", "set_experiment", "start_run"]
    assert calls[0][1][0].endswith("/mlruns")
    import os

    assert os.environ["MLFLOW_ALLOW_FILE_STORE"] == "true"
    logged = dict(calls)
    assert logged["log_metrics"][0] == {"faithfulness": 0.889, "n": 14.0}
    assert [a[1] for n, a in calls if n == "log_dict"] == ["meta.json", "by_category.json"]


def test_real_mlflow_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    run_id = obs.log_eval_run(META, METRICS, kind="answer-eval")
    run = mlflow.tracking.MlflowClient().get_run(run_id)
    assert run.data.metrics == {"faithfulness": 0.889, "n": 14.0}
    assert run.data.params["agent.agent_model"] == "qwen3:4b"
    assert run.data.tags["kind"] == "answer-eval"
