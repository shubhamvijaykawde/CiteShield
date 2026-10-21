"""Deterministic stand-ins for the GPU models and the LLM, shared by unit tests."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from citeshield.agent.llm import LLMResponse, ToolCall, Usage
from citeshield.contracts.chunk import (
    ChunkKind,
    ChunkRecord,
    TrustTier,
    estimate_tokens,
    make_chunk_id,
    sha256_text,
)
from citeshield.contracts.verification import Entailment
from citeshield.retrieval.embed import EncodedText

DIM = 64


class HashingEncoder:
    """Bag-of-words hashed into a dense vector and a sparse vector."""

    dim = DIM

    def encode(self, texts: list[str]) -> list[EncodedText]:
        out = []
        for text in texts:
            dense = [0.0] * DIM
            sparse: dict[int, float] = {}
            for tok in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(tok.encode(), usedforsecurity=False).hexdigest(), 16)
                dense[h % DIM] += 1.0
                sparse[h % 50_000] = 1.0
            norm = math.sqrt(sum(x * x for x in dense)) or 1.0
            out.append(EncodedText([x / norm for x in dense], sparse))
        return out


class KeywordReranker:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [1.0 if self.keyword in p else 0.1 for p in passages]


def make_chunk(
    regulation: str,
    article: int,
    text: str,
    *,
    point: str | None = None,
    paragraph: int | None = None,
    heading: str = "Article",
) -> ChunkRecord:
    celex, version = "32099R0002", "v1"
    return ChunkRecord(
        chunk_id=make_chunk_id(celex=celex, kind=ChunkKind.ARTICLE, version=version,
                               article=article, paragraph=paragraph, point=point),
        regulation=regulation, celex=celex, kind=ChunkKind.ARTICLE, article=article,
        paragraph=paragraph, point=point, heading_path=[regulation, heading], text=text,
        source_url="https://eur-lex.europa.eu/x", version=version, version_date=date(2099, 1, 1),
        content_sha256=sha256_text(text), n_tokens=estimate_tokens(text),
        trust_tier=TrustTier.OFFICIAL,
    )  # fmt: skip


_ids = itertools.count(1)


def text_turn(text: str) -> LLMResponse:
    return LLMResponse(
        content=[{"type": "text", "text": text}],
        text=text,
        tool_calls=[],
        stop_reason="end_turn",
        model="fake",
        usage=Usage(input_tokens=10, output_tokens=5, calls=1),
    )


def json_turn(payload: dict[str, Any]) -> LLMResponse:
    return text_turn(json.dumps(payload))


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    tcs = [ToolCall(f"toolu_{next(_ids)}", name, args) for name, args in calls]
    return LLMResponse(
        content=[{"type": "tool_use", "id": t.id, "name": t.name, "input": t.input} for t in tcs],
        text="",
        tool_calls=tcs,
        stop_reason="tool_use",
        model="fake",
        usage=Usage(input_tokens=10, output_tokens=5, calls=1),
    )


@dataclass
class ScriptedLLM:
    """Returns scripted responses per model; records every request it receives."""

    guard_label: str = "in_scope"
    agent_turns: list[LLMResponse] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if kwargs["model"] == "guard-model":
            return json_turn({"label": self.guard_label})
        if not self.agent_turns:
            raise AssertionError("ScriptedLLM ran out of agent turns")
        return self.agent_turns.pop(0)


class EntailAllNLI:
    """Stand-in NLI that entails every pair.

    For tests of graph mechanics that predate verification: the deterministic citation and
    quote checks still run for real; only the entailment model is replaced.
    """

    name = "test/entail-all"
    max_tokens = 512

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def score(self, pairs: Any) -> list[Entailment]:
        return [Entailment(entailment=1.0, neutral=0.0, contradiction=0.0) for _ in pairs]


class RuleNLI:
    """Deterministic NLI double driven by substring rules on the hypothesis.

    ``rules`` maps a substring of the hypothesis to (entail, neutral, contradict); the first
    rule whose key appears wins. Pairs whose premise contains ``contradicting_premise`` get
    the contradiction override instead (to model a source that disagrees).
    """

    name = "test/rules"
    max_tokens = 512

    def __init__(
        self,
        rules: dict[str, tuple[float, float, float]],
        default: tuple[float, float, float] = (0.05, 0.9, 0.05),
        contradicting_premise: str | None = None,
    ) -> None:
        self.rules = rules
        self.default = default
        self.contradicting_premise = contradicting_premise
        self.pairs: list[tuple[str, str]] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def score(self, pairs: Any) -> list[Entailment]:
        out = []
        for premise, hypothesis in pairs:
            self.pairs.append((premise, hypothesis))
            probs = next((v for k, v in self.rules.items() if k in hypothesis), self.default)
            if self.contradicting_premise and self.contradicting_premise in premise:
                probs = (0.02, 0.03, 0.95)
            e, n, c = probs
            out.append(Entailment(entailment=e, neutral=n, contradiction=c))
        return out


def entail_all(executor: Any) -> Any:
    """A real Verifier over the executor's corpus with the entail-all NLI stand-in."""
    from citeshield.verify.verifier import Verifier

    return Verifier(executor.corpus, EntailAllNLI())
