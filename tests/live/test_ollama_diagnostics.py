"""Isolated live diagnostics for qwen3:4b on Ollama, one capability per test.

These use the small fixture corpus and a hashing encoder, so they need only the Ollama server
(no GPU index or reranker). Each test checks one thing the agent relies on:

* grammar: the draft schema makes an uncited answer impossible (an application guarantee);
* structured output: the real draft JSON schema -> a valid AgentAnswer citing given sources;
* tool calling: a tool call the allowlisted executor accepts;
* repair: after a rejected draft, the model returns a corrected answer that passes validation.

All but the grammar test are marked ``model_quality``: they measure how reliably qwen3:4b
makes the right choice, so an occasional failure is a model-reliability signal, not proof of
a code defect. Run only the guarantees with ``-m "live and not model_quality"``.

    py -m uv run pytest -m live tests/live/test_ollama_diagnostics.py -s
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
from qdrant_client import QdrantClient

from citeshield.agent import prompts
from citeshield.agent.config import AgentConfig
from citeshield.agent.graph import CiteShieldAgent, citation_errors
from citeshield.agent.llm import LLMResponse
from citeshield.agent.ollama import OllamaLLM
from citeshield.agent.schema import draft_schema
from citeshield.agent.tools import ALLOWED_TOOLS, EvidenceItem, ToolExecutor, format_sources
from citeshield.config import IngestConfig
from citeshield.contracts.answer import AgentAnswer, AnswerStatus
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.validate import validate_document
from citeshield.retrieval.config import RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.expand import Expander
from citeshield.retrieval.hybrid import HybridSearcher
from citeshield.retrieval.index import build_index
from citeshield.retrieval.pipeline import Retriever
from citeshield.verify.config import VerifierConfig
from citeshield.verify.nli import DebertaNLI
from citeshield.verify.verifier import Verifier
from helpers import HashingEncoder, json_turn, text_turn, tool_turn

pytestmark = pytest.mark.live

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b"
P2 = "32099R0001:art2:p2@oj-2099-01-01"
SYSTEM = "You answer questions about the Test Regulation using only the sources provided."


@pytest.fixture(scope="module")
def ollama(local_network_only: list[str]) -> OllamaLLM:
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError) as e:
        pytest.skip(f"Ollama server not reachable at {OLLAMA_URL} ({type(e).__name__})")
    if MODEL not in {m["name"] for m in tags.get("models", [])}:
        pytest.skip(f"model {MODEL} is not pulled (run: ollama pull {MODEL})")
    return OllamaLLM(OLLAMA_URL)


@pytest.fixture
def executor(parsed: ParseResult) -> ToolExecutor:
    corpus = Corpus(validate_document(parsed.records, parsed.toc, IngestConfig()).valid)
    cfg = RetrieverConfig(collection="diag")
    client = QdrantClient(":memory:")
    build_index(client, corpus, HashingEncoder(), cfg)
    searcher = HybridSearcher(client, corpus, HashingEncoder(), cfg)
    return ToolExecutor(Retriever(searcher, None, Expander(corpus), cfg))


def _show(name: str, resp: LLMResponse) -> None:
    keep = ("finish_reason", "content_chars", "thinking_chars", "tool_calls", "output_tokens")
    print(f"\n[{name}] {json.dumps({k: resp.meta.get(k) for k in keep})}")


_CONTENT_FREE_ANSWER = (
    'Output exactly this JSON and nothing else: {"status": "answered", "summary": "Yes.", '
    '"claims": [], "checklist": [], "refusal_reason": null, "caveats": []}'
)


def test_draft_grammar_forbids_uncited_answers(
    ollama: OllamaLLM, local_network_only: list[str]
) -> None:
    """Application guarantee, not model quality: Ollama's grammar-constrained decoding must
    make an answer with nothing cited impossible, even when the prompt asks for exactly that.
    Only rules the schema encodes are asserted; content quality is Pydantic's and the
    verifier's job."""
    assert OllamaLLM.schema_status_rules
    resp = ollama.chat(
        model=MODEL,
        system="Return JSON.",
        messages=[{"role": "user", "content": _CONTENT_FREE_ANSWER}],
        json_schema=draft_schema(status_rules=True),
    )
    _show("grammar", resp)
    assert resp.meta["finish_reason"] == "stop", "draft was cut off (thinking used the budget)"
    draft = resp.json()
    if draft["status"] == "refused":
        assert draft["refusal_reason"] is not None
        assert draft["claims"] == draft["checklist"] == []
    else:
        assert draft["claims"] or draft["checklist"], f"uncited {draft['status']} answer"
    for entry in draft["claims"] + draft["checklist"]:
        assert entry["citations"], "claim or checklist item without a citation"
    assert local_network_only == []


@pytest.mark.model_quality
def test_structured_output_with_answer_schema(
    ollama: OllamaLLM, executor: ToolExecutor, local_network_only: list[str]
) -> None:
    """Model reliability: qwen3:4b must choose to answer and cite correctly. The schema keeps
    the shape valid, but which status the model picks is its own call."""
    chunk = executor.corpus[P2]
    evidence = {P2: EvidenceItem.from_chunk(chunk, "test")}
    user = (
        "<question>\nDoes paragraph 1 of Article 2 apply to widgets used for research?\n"
        f"</question>\n\nSources:\n{format_sources(list(evidence.values()))}\n\n"
        + prompts.load("draft").text
    )
    resp = ollama.chat(
        model=MODEL,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        json_schema=draft_schema(status_rules=OllamaLLM.schema_status_rules),
    )
    _show("structured", resp)
    assert resp.meta["finish_reason"] == "stop", "draft was cut off (thinking used the budget)"
    answer = AgentAnswer.model_validate(
        {**resp.json(), "checklist": resp.json()["checklist"] or None}
    )
    assert citation_errors(answer, evidence) == []
    assert answer.status is AnswerStatus.ANSWERED
    assert local_network_only == []


@pytest.mark.model_quality
def test_tool_calling(
    ollama: OllamaLLM, executor: ToolExecutor, local_network_only: list[str]
) -> None:
    resp = ollama.chat(
        model=MODEL,
        system=SYSTEM + " Use the tools to read the regulation before answering.",
        messages=[
            {"role": "user", "content": "What does paragraph 2 of Article 2 of Test Reg say?"}
        ],
        tools=executor.specs(),
    )
    _show("tools", resp)
    assert resp.tool_calls, f"no tool call; finish_reason={resp.meta['finish_reason']}"
    for call in resp.tool_calls:
        assert call.name in ALLOWED_TOOLS
        outcome = executor.execute(call.name, call.input)
        assert not outcome.is_error, outcome.content
    assert local_network_only == []


class _ScriptThenOllama:
    """Scripted turns first (to set up a rejected draft), then the real model."""

    provider = "scripted+ollama"
    schema_status_rules = OllamaLLM.schema_status_rules

    def __init__(self, scripted: list[LLMResponse], real: OllamaLLM) -> None:
        self.scripted = scripted
        self.real = real

    def chat(self, **kwargs: Any) -> LLMResponse:
        if self.scripted:
            return self.scripted.pop(0)
        return self.real.chat(**kwargs)


@pytest.mark.model_quality
def test_repair_path(
    ollama: OllamaLLM, executor: ToolExecutor, local_network_only: list[str]
) -> None:
    bogus_draft = {
        "status": "answered",
        "summary": "Research widgets are exempt under GDPR Art. 22.",
        "claims": [
            {
                "text": "Research widgets are exempt.",
                "citations": [
                    {
                        "chunk_id": "32016R0679:art22:p1@oj-2016-05-04",
                        "quote": "solely for research",
                    }
                ],
            }
        ],
        "checklist": [],
        "refusal_reason": None,
        "caveats": [],
    }
    scripted = [
        json_turn({"label": "in_scope"}),
        tool_turn(("get_article", {"regulation": "Test Reg", "article": 2, "paragraph": 2})),
        text_turn("Article 2(2) exempts widgets used solely for research."),
        json_turn(bogus_draft),  # cites a chunk no tool returned -> must be rejected
    ]
    llm = _ScriptThenOllama(scripted, ollama)
    # The real verifier (DeBERTa NLI, offline, CPU): the repaired draft must also verify.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    vcfg = VerifierConfig()
    nli = DebertaNLI(vcfg.nli_model, revision=vcfg.nli_revision, device=vcfg.device)
    cfg = AgentConfig(agent_model=MODEL, guard_model=MODEL)
    agent = CiteShieldAgent(llm, executor, cfg, verifier=Verifier(executor.corpus, nli, vcfg))
    run = agent.run("Are widgets used solely for research exempt from Article 2(1)?")

    repair = run.llm_log[-1]
    keys = ("provider", "finish_reason", "content_chars", "thinking_chars", "output_tokens")
    print(
        f"\n[repair] status={run.answer.status.value} rejections={run.draft_rejections} "
        f"repair_call={json.dumps({k: repair.get(k) for k in keys})}"
    )
    assert len(run.draft_rejections) == 1
    assert "was not returned by any tool" in run.draft_rejections[0][0]
    assert repair["provider"] == "ollama" and repair["stage"] == "draft"
    assert run.answer.status is AnswerStatus.ANSWERED, run.answer.caveats
    assert {c.chunk_id for c in run.answer.all_citations()} <= set(run.evidence)
    assert local_network_only == []
