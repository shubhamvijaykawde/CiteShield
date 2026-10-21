"""Provider selection and the Ollama adapter, offline (httpx.MockTransport, no server)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from citeshield.agent import factory
from citeshield.agent.config import AgentConfig, LLMSettings
from citeshield.agent.graph import CiteShieldAgent
from citeshield.agent.llm import AnthropicLLM
from citeshield.agent.ollama import (
    LLMContextExceeded,
    OllamaLLM,
    from_ollama_response,
    is_loopback,
    to_ollama_messages,
    to_ollama_tool,
)
from citeshield.agent.tools import ToolExecutor
from citeshield.config import IngestConfig, Settings
from citeshield.contracts.answer import AnswerStatus
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.validate import validate_document
from citeshield.retrieval.config import RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.expand import Expander
from citeshield.retrieval.hybrid import HybridSearcher
from citeshield.retrieval.index import build_index
from citeshield.retrieval.pipeline import Retriever
from helpers import HashingEncoder

P2 = "32099R0001:art2:p2@oj-2099-01-01"


def settings(**llm: Any) -> Settings:
    return Settings(_env_file=None, llm=LLMSettings(**llm))  # type: ignore[call-arg]


# ------------------------------------------------------------------------ selection


def test_default_provider_is_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CITESHIELD_LLM__PROVIDER", raising=False)
    assert Settings(_env_file=None).llm.provider == "anthropic"  # type: ignore[call-arg]
    llm, cfg = factory.build_llm(settings())
    assert isinstance(llm, AnthropicLLM)
    assert cfg.agent_model == "claude-sonnet-5-5"


def test_ollama_provider_never_constructs_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("Anthropic client constructed while provider=ollama")

    monkeypatch.setattr(factory, "AnthropicLLM", forbidden)
    llm, cfg = factory.build_llm(settings(provider="ollama", ollama_num_ctx=8192))
    assert isinstance(llm, OllamaLLM)
    assert llm.provider == "ollama" and llm.num_ctx == 8192
    assert (cfg.agent_model, cfg.guard_model) == ("qwen3:4b", "qwen3:4b")
    # Everything else in the agent config is kept.
    assert cfg.max_tool_rounds == AgentConfig().max_tool_rounds


def test_provider_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITESHIELD_LLM__PROVIDER", "ollama")
    monkeypatch.setenv("CITESHIELD_LLM__OLLAMA_MODEL", "qwen3:8b")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert (s.llm.provider, s.llm.ollama_model) == ("ollama", "qwen3:8b")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMSettings(provider="openai")  # type: ignore[arg-type]


def test_loopback_detection() -> None:
    assert is_loopback("http://localhost:11434")
    assert is_loopback("http://127.0.0.1:11434")
    assert not is_loopback("https://api.example.com")


# ------------------------------------------------------------------------ translation


def test_history_translation() -> None:
    history = [
        {"role": "user", "content": "<question>q</question>"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "id": "t1", "name": "retrieve", "input": {"query": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "<source>..</source>"},
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "Error: x",
                    "is_error": True,
                },
            ],
        },
        {"role": "user", "content": "Write the answer."},
    ]
    out = to_ollama_messages("SYS", history)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2] == {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [{"id": "t1", "function": {"name": "retrieve", "arguments": {"query": "x"}}}],
    }
    assert [m["role"] for m in out[3:]] == ["tool", "tool", "user"]
    assert out[3]["tool_name"] == "retrieve"


def test_unknown_block_type_is_an_error() -> None:
    with pytest.raises(ValueError, match="no Ollama equivalent"):
        to_ollama_messages("S", [{"role": "user", "content": [{"type": "image"}]}])


def test_tool_spec_translation() -> None:
    spec = {
        "name": "retrieve",
        "description": "d",
        "strict": True,
        "input_schema": {"type": "object"},
    }
    assert to_ollama_tool(spec) == {
        "type": "function",
        "function": {"name": "retrieve", "description": "d", "parameters": {"type": "object"}},
    }


@pytest.mark.parametrize("args", [{"article": 2}, '{"article": 2}'])
def test_response_parsing(args: Any) -> None:
    data = {
        "model": "qwen3:4b",
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "get_article", "arguments": args}}],
        },
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    }
    r = from_ollama_response(data, "qwen3:4b")
    assert r.stop_reason == "tool_use"
    assert r.tool_calls[0].input == {"article": 2} and r.tool_calls[0].id.startswith("call_")
    assert r.content[0]["type"] == "tool_use"
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cost_usd) == (100, 20, 0.0)

    truncated = from_ollama_response({"message": {"content": "{"}, "done_reason": "length"}, "m")
    assert truncated.stop_reason == "max_tokens"


def test_oversized_prompt_is_refused_not_truncated() -> None:
    def never(_req: httpx.Request) -> httpx.Response:
        raise AssertionError("request should not be sent")

    llm = OllamaLLM(
        num_ctx=2048,
        num_predict=256,
        client=httpx.Client(
            base_url="http://localhost:11434", transport=httpx.MockTransport(never)
        ),
    )
    with pytest.raises(LLMContextExceeded, match="truncate silently"):
        llm.chat(model="m", system="x" * 10_000, messages=[{"role": "user", "content": "q"}])


# ------------------------------------------------------------------------ graph via adapter


@pytest.fixture
def executor(parsed: ParseResult) -> ToolExecutor:
    corpus = Corpus(validate_document(parsed.records, parsed.toc, IngestConfig()).valid)
    cfg = RetrieverConfig(collection="test")
    client = QdrantClient(":memory:")
    build_index(client, corpus, HashingEncoder(), cfg)
    return ToolExecutor(
        Retriever(
            HybridSearcher(client, corpus, HashingEncoder(), cfg), None, Expander(corpus), cfg
        )
    )


def _reply(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"model": "qwen3:4b", "message": msg, "done": True, "done_reason": "stop",
            "prompt_eval_count": 50, "eval_count": 10}  # fmt: skip


def test_full_graph_runs_through_the_ollama_adapter(executor: ToolExecutor) -> None:
    draft = {
        "status": "answered",
        "summary": "Test Reg Art. 2(2) exempts research widgets.",
        "claims": [
            {
                "text": "Research widgets are exempt.",
                "citations": [{"chunk_id": P2, "quote": "shall not apply to widgets used solely"}],
            }
        ],
        "checklist": [],
        "refusal_reason": None,
        "caveats": [],
    }
    script = [
        _reply('{"label": "in_scope"}'),
        _reply(
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "get_article",
                        "arguments": {"regulation": "Test Reg", "article": 2, "paragraph": 2},
                    },
                }
            ]
        ),
        _reply("Found Art. 2(2)."),
        _reply(json.dumps(draft)),
    ]
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "localhost" and request.url.path == "/api/chat"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=script.pop(0))

    llm = OllamaLLM(
        client=httpx.Client(
            base_url="http://localhost:11434", transport=httpx.MockTransport(handler)
        )
    )
    cfg = AgentConfig(agent_model="qwen3:4b", guard_model="qwen3:4b")
    run = CiteShieldAgent(llm, executor, cfg).run("Are research widgets exempt?")

    assert run.answer.status is AnswerStatus.ANSWERED
    assert run.meta["llm_provider"] == "ollama"
    guard, research1, research2, draft_req = bodies
    assert "format" in guard and "tools" not in guard
    assert [t["function"]["name"] for t in research1["tools"]] == [
        "retrieve", "get_article", "compare_articles", "generate_checklist",
    ]  # fmt: skip
    tool_msg = research2["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_name"] == "get_article"
    assert "<source" in tool_msg["content"]
    # The draft forbids tool calls by omitting tools, and is grammar-constrained to the schema.
    assert "tools" not in draft_req and draft_req["format"]["required"][0] == "status"
    assert all(b["options"]["num_ctx"] == 16384 and b["think"] is True for b in bodies)
    assert run.usage.cost_usd == 0.0
