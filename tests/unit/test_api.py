"""Local web app: UI, readiness, limits, modes and refusals, with scripted agents (no models)."""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from citeshield.agent.config import AgentConfig
from citeshield.agent.graph import CiteShieldAgent
from citeshield.agent.llm import LLMResponse
from citeshield.agent.tools import ToolExecutor
from citeshield.api.main import STATIC_DIR, create_app
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
READ_P2 = ("get_article", {"regulation": "Test Reg", "article": 2, "paragraph": 2})
DIAG = {"provider": "ollama", "model": "qwen3:4b", "paid": False, "reachable": True}
DRAFT = {
    "status": "answered",
    "summary": "Under Test Reg Art. 2(2), research widgets are exempt from paragraph 1.",
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


def _turns() -> list[LLMResponse]:
    return [tool_turn(READ_P2), text_turn("ok"), json_turn(DRAFT)]


def _factory(executor: ToolExecutor, llm: ScriptedLLM) -> Callable[[], CiteShieldAgent]:
    return lambda: CiteShieldAgent(llm, executor, CFG, verifier=entail_all(executor))


def _events(body: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for block in body.strip().split("\n\n"):
        kind = re.search(r"^event: (.*)$", block, re.M)
        data = re.search(r"^data: (.*)$", block, re.M)
        if kind and data:
            out.append((kind.group(1), json.loads(data.group(1))))
    return out


# ------------------------------------------------------------------------ UI


def test_browser_ui_is_served_without_external_resources(executor: ToolExecutor) -> None:
    app = create_app(_factory(executor, ScriptedLLM()), diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "text/html" in page.headers["content-type"]
        assert "not legal advice" in page.text.lower()
        assert page.headers["cache-control"] == "no-cache"  # an updated UI is never stale
        for asset in ("/static/app.js", "/static/style.css"):
            r = client.get(asset)
            assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        assert "cache-control" not in client.get("/health").headers
    for f in STATIC_DIR.iterdir():
        text = f.read_text("utf-8")
        # No CDN or other network resource: everything is served locally.
        assert not re.search(r"""(src|href)=["']https?://""", text), f.name
        assert "@import" not in text and "cdn" not in text.lower(), f.name


def test_ui_never_renders_model_text_as_html() -> None:
    js = (STATIC_DIR / "app.js").read_text("utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js


def test_ui_examples_match_the_golden_set() -> None:
    from citeshield.config import PROJECT_ROOT
    from citeshield.evaluation.golden import load_golden

    questions = {g.question: g for g in load_golden(PROJECT_ROOT / "eval/golden/questions.jsonl")}
    js = (STATIC_DIR / "app.js").read_text("utf-8")
    examples = re.findall(r'q: "([^"]+)"', js)
    assert len(examples) == 5
    in_golden = [q for q in examples if q in questions]
    # Four are golden questions with reviewed or checked expected citations; the breach
    # deadline question is the live-test question (GDPR Art. 33(1)).
    assert len(in_golden) >= 4
    assert any(questions[q].should_refuse for q in in_golden)


# ------------------------------------------------------------------------ readiness


def test_health_and_ready_while_loading_and_after(executor: ToolExecutor) -> None:
    gate = threading.Event()

    def slow() -> CiteShieldAgent:
        gate.wait(5)
        return _factory(executor, ScriptedLLM())()

    app = create_app(slow, load_in_background=True, diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["status"] == "loading" and h["llm"] == DIAG and h["tracing"] == "off"
        assert client.get("/ready").status_code == 503
        assert client.post("/ask", json={"question": "Are widgets exempt?"}).status_code == 503
        assert client.get("/").status_code == 200  # the page works while loading
        gate.set()
        for _ in range(100):
            if client.get("/ready").status_code == 200:
                break
            time.sleep(0.05)
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["regulations"] == ["Test Reg"]


def test_startup_failure_is_reported_not_hidden(executor: ToolExecutor) -> None:
    def broken() -> CiteShieldAgent:
        raise FileNotFoundError("data/index/manifest.json: run citeshield-index")

    app = create_app(broken, diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        h = client.get("/health").json()
        assert h["status"] == "error" and "citeshield-index" in h["error"]
        assert client.get("/ready").status_code == 503
        r = client.post("/ask", json={"question": "Are widgets exempt?"})
        assert r.status_code == 503 and "citeshield-index" in r.json()["detail"]


# ------------------------------------------------------------------------ limits


def test_one_question_at_a_time(executor: ToolExecutor) -> None:
    app = create_app(
        _factory(executor, ScriptedLLM(agent_turns=_turns())), diagnostics=lambda: DIAG
    )
    with TestClient(app) as client:
        app.state.busy.acquire()
        r = client.post("/ask", json={"question": "Are widgets exempt?"})
        assert r.status_code == 429
        app.state.busy.release()
        events = _events(client.post("/ask", json={"question": "Are widgets exempt?"}).text)
        assert events[-1][0] == "final"
        assert not app.state.busy.locked()  # released after the run


def test_request_body_limit(executor: ToolExecutor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITESHIELD_API_MAX_BODY", "200")
    app = create_app(_factory(executor, ScriptedLLM()), diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "x" * 500})
        assert r.status_code == 413


def test_answer_timeout_is_reported(
    executor: ToolExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CITESHIELD_API_TIMEOUT_S", "0.3")

    class Slow(ScriptedLLM):
        def chat(self, **kwargs: Any) -> LLMResponse:
            if kwargs["model"] == "agent-model":
                time.sleep(0.6)
            return super().chat(**kwargs)

    app = create_app(_factory(executor, Slow(agent_turns=_turns())), diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        events = _events(client.post("/ask", json={"question": "Are widgets exempt?"}).text)
        assert events[-1][0] == "error" and "no answer within" in events[-1][1]["message"]
        for _ in range(100):  # the run finishes in the background, then frees the slot
            if not app.state.busy.locked():
                break
            time.sleep(0.05)
        assert not app.state.busy.locked()


# ------------------------------------------------------------------------ modes and outputs


@pytest.mark.parametrize(
    ("mode", "hint"),
    [("answer", None), ("checklist", "compliance checklist"), ("compare", "compare provisions")],
)
def test_mode_reaches_the_agent(executor: ToolExecutor, mode: str, hint: str | None) -> None:
    llm = ScriptedLLM(agent_turns=_turns())
    app = create_app(_factory(executor, llm), diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        events = _events(
            client.post("/ask", json={"question": "Are widgets exempt?", "mode": mode}).text
        )
    assert events[-1][0] == "final"
    first = next(c for c in llm.calls if c["model"] == "agent-model")
    question = first["messages"][0]["content"]
    if hint is None:
        assert question.endswith("</question>")
    else:
        assert hint in question
    with TestClient(create_app(_factory(executor, ScriptedLLM()), diagnostics=lambda: DIAG)) as c:
        assert c.post("/ask", json={"question": "Q here", "mode": "essay"}).status_code == 422


def test_final_event_carries_claims_sources_and_verification(executor: ToolExecutor) -> None:
    app = create_app(
        _factory(executor, ScriptedLLM(agent_turns=_turns())), diagnostics=lambda: DIAG
    )
    with TestClient(app) as client:
        events = _events(client.post("/ask", json={"question": "Are widgets exempt?"}).text)
        final = events[-1][1]
        assert final["answer"]["claims"][0]["citations"][0]["chunk_id"] == P2
        assert final["sources"][P2]["label"] == "Test Reg Art. 2(2)"
        assert final["verification"]["claims"][0]["verdict"] == "supported"
        src = client.get(f"/sources/{P2}").json()
        assert src["celex"] == "32099R0001" and src["version"] == "oj-2099-01-01"


def test_refusal_is_explained(executor: ToolExecutor) -> None:
    llm = ScriptedLLM(guard_label="other_law")
    app = create_app(_factory(executor, llm), diagnostics=lambda: DIAG)
    with TestClient(app) as client:
        events = _events(client.post("/ask", json={"question": "Irish corporate tax?"}).text)
    answer = events[-1][1]["answer"]
    assert answer["status"] == "refused" and answer["refusal_reason"] == "out_of_scope"
    assert "outside the EU texts I cover" in answer["summary"]


def test_ui_explains_every_refusal_reason() -> None:
    from citeshield.contracts.answer import RefusalReason

    js = (STATIC_DIR / "app.js").read_text("utf-8")
    for reason in RefusalReason:
        assert f"{reason.value}:" in js, reason


def test_static_dir_ships_with_the_package() -> None:
    assert {p.name for p in STATIC_DIR.iterdir()} >= {"index.html", "app.js", "style.css"}
    assert STATIC_DIR.parent == Path(__file__).parents[2] / "src" / "citeshield" / "api"
