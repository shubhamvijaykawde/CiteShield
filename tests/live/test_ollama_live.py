"""Live integration test: the real agent (GPU retriever + local qwen3:4b via Ollama).

Opt-in and slow (minutes per question):

    py -m uv run pytest -m live tests/live -s

Asserted: the pipeline's invariants (valid AgentAnswer, citations only to retrieved chunks,
allowlist respected, only local network traffic, no paid provider constructed).
Recorded, not asserted: answer quality (status, whether the expected article was cited,
whether quotes are verbatim). That is what milestones 5-6 measure; a 4B model is expected to
be weaker, and the results file shows how.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from citeshield.agent import factory
from citeshield.agent.config import LLMSettings
from citeshield.agent.graph import AgentRun, CiteShieldAgent
from citeshield.agent.ollama import OllamaLLM, is_loopback
from citeshield.agent.tools import ALLOWED_TOOLS
from citeshield.api.main import create_app
from citeshield.config import PROJECT_ROOT, Settings
from citeshield.contracts.answer import AgentAnswer
from citeshield.evaluation.golden import parse_ref
from citeshield.retrieval.corpus import Corpus

pytestmark = pytest.mark.live

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b"
RESULTS = PROJECT_ROOT / "eval" / "results" / "live_ollama_latest.json"

CASES = {
    "answerable": {
        "question": "Within how many hours must a controller notify a personal data breach "
        "to the supervisory authority?",
        "expected": ["GDPR 33(1)"],
    },
    "unanswerable": {
        "question": "What is the corporate income tax rate in Ireland?",
        "expected": [],
    },
}
_records: dict[str, Any] = {}


def _skip_reason() -> str | None:
    try:
        version = httpx.get(f"{OLLAMA_URL}/api/version", timeout=3).json()["version"]
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return f"Ollama server not reachable at {OLLAMA_URL} ({type(e).__name__})"
    if MODEL not in {m["name"] for m in tags.get("models", [])}:
        return f"model {MODEL} is not pulled on Ollama {version} (run: ollama pull {MODEL})"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    if not (settings.index_dir / "manifest.json").exists() or not settings.chunks_path.exists():
        return "retrieval index not built (run citeshield-ingest and citeshield-index)"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return "ml extra not installed (uv sync --extra ml)"
    return None


@pytest.fixture(scope="module")
def local_agent(hf_offline: None, local_network_only: list[str]) -> Iterator[CiteShieldAgent]:
    if reason := _skip_reason():
        pytest.skip(reason)

    original_anthropic = factory.AnthropicLLM

    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("paid provider (AnthropicLLM) constructed during the local test")

    factory.AnthropicLLM = forbidden  # type: ignore[misc,assignment]
    try:
        # Explicit provider; .env is ignored so a local .env cannot redirect the test.
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            llm=LLMSettings(provider="ollama", ollama_base_url=OLLAMA_URL, ollama_model=MODEL),
        )
        agent = factory.build_agent(settings)
        yield agent
    finally:
        factory.AnthropicLLM = original_anthropic  # type: ignore[misc]
        if _records:
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            RESULTS.write_text(json.dumps(_records, indent=2, ensure_ascii=False), "utf-8")
            print(f"\nlive results written to {RESULTS}")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _assert_invariants(run: AgentRun, blocked: list[str]) -> None:
    AgentAnswer.model_validate(run.answer.model_dump())  # still satisfies the contract
    for citation in run.answer.all_citations():
        assert citation.chunk_id in run.evidence, f"cited unretrieved chunk {citation.chunk_id}"
    for call in run.tool_log:
        assert call["name"] in ALLOWED_TOOLS or call["is_error"]
    assert run.meta["llm_provider"] == "ollama"
    assert blocked == [], f"non-local connections attempted: {blocked}"


def _record(
    name: str, run: AgentRun, corpus: Corpus, elapsed: float, expected: list[str]
) -> dict[str, Any]:
    cited_ids = list(run.cited_sources())
    refs = [parse_ref(r) for r in expected]
    # Diagnostic only: the claim-level verifier (milestone 5) is not built yet, so nothing
    # in the pipeline checks quotes. This records how often the quote is really verbatim.
    quotes = [
        (c.quote, run.evidence[c.chunk_id].text)
        for c in run.answer.all_citations()
        if c.chunk_id in run.evidence
    ]
    verbatim = sum(1 for q, src in quotes if _norm(q) in _norm(src))
    rec = {
        "question": CASES[name]["question"],
        "status": run.answer.status.value,
        "refusal_reason": run.answer.refusal_reason.value if run.answer.refusal_reason else None,
        "guard_label": run.guard_label,
        "summary": run.answer.summary,
        "claims": [c.text for c in run.answer.claims],
        "cited": [run.evidence[i].label for i in cited_ids],
        "expected": expected,
        "expected_cited": [r.label for r in refs if any(r.matches(corpus[i]) for i in cited_ids)],
        "quotes_verbatim": f"{verbatim}/{len(quotes)}",
        "tool_calls": [(t["name"], t["input"], t["is_error"]) for t in run.tool_log],
        "security_flags": run.security_flags,
        "caveats": run.answer.caveats,
        "llm_calls": run.usage.calls,
        "tokens_in_out": [run.usage.input_tokens, run.usage.output_tokens],
        "cost_usd": run.usage.cost_usd,
        "seconds": round(elapsed, 1),
    }
    _records[name] = rec
    print(f"\n[{name}] {json.dumps(rec, indent=2, ensure_ascii=False)}")
    return rec


def test_wiring_is_local(local_agent: CiteShieldAgent) -> None:
    assert isinstance(local_agent.llm, OllamaLLM)
    assert is_loopback(local_agent.llm.base_url)
    assert local_agent.llm.num_ctx >= 16384
    assert (local_agent.cfg.agent_model, local_agent.cfg.guard_model) == (MODEL, MODEL)


@pytest.mark.parametrize("name", list(CASES))
def test_agent_question(
    local_agent: CiteShieldAgent, local_network_only: list[str], name: str
) -> None:
    case = CASES[name]
    t0 = time.perf_counter()
    run = local_agent.run(case["question"])
    rec = _record(name, run, local_agent.tools.corpus, time.perf_counter() - t0, case["expected"])
    _assert_invariants(run, local_network_only)
    assert rec["cost_usd"] == 0.0


def test_sse_endpoint(local_agent: CiteShieldAgent, local_network_only: list[str]) -> None:
    app = create_app(lambda: local_agent)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["llm_provider"] == "ollama"
        resp = client.post(
            "/ask", json={"question": "Does the California Consumer Privacy Act require a "
                          "'Do Not Sell My Personal Information' link?"},
        )  # fmt: skip
    assert resp.status_code == 200
    events = []
    for block in resp.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    kinds = [k for k, _ in events]
    _records["sse"] = {"event_kinds": kinds, "final": events[-1][1].get("answer")}
    print(f"\n[sse] events: {kinds}\n[sse] answer: {json.dumps(events[-1][1].get('answer'))}")
    assert kinds[0] == "status" and kinds[-1] == "final", kinds
    AgentAnswer.model_validate(events[-1][1]["answer"])
    assert local_network_only == []
