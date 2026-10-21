"""The agent's tools: read-only, allowlisted, argument-validated.

Every chunk a tool returns is recorded in the evidence registry. The draft may only cite
chunk ids from that registry, which is what makes "cite only what you retrieved" checkable.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from citeshield.contracts.chunk import ChunkKind, ChunkRecord
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.hybrid import Hit
from citeshield.retrieval.pipeline import Retriever

ALLOWED_TOOLS = ("retrieve", "get_article", "compare_articles", "generate_checklist")

_OBLIGATION = re.compile(r"\bshall\b")
_DELIMITER = re.compile(r"</?\s*source\b", re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    label: str
    regulation: str
    kind: str
    text: str
    source_url: str
    origin: str  # which tool / retrieval path produced it
    score: float | None = None

    @classmethod
    def from_chunk(cls, c: ChunkRecord, origin: str, score: float | None = None) -> EvidenceItem:
        return cls(
            chunk_id=c.chunk_id,
            label=c.citation_label,
            regulation=c.regulation,
            kind=c.kind.value,
            text=c.text,
            source_url=str(c.source_url),
            origin=origin,
            score=score,
        )


@dataclass
class ToolOutcome:
    content: str
    is_error: bool = False
    evidence: list[EvidenceItem] = field(default_factory=list)
    security_flag: str | None = None
    elapsed_ms: float = 0.0


# --------------------------------------------------------------------------- arguments


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrieveArgs(_Args):
    query: str = Field(min_length=3, max_length=500)
    regulations: list[str] = Field(max_length=5)


class GetArticleArgs(_Args):
    regulation: str
    article: int = Field(ge=1, le=300)
    paragraph: int = Field(ge=0, le=60, description="0 = the whole article")


class ArticleRef(_Args):
    regulation: str
    article: int = Field(ge=1, le=300)


class CompareArgs(_Args):
    left: ArticleRef
    right: ArticleRef
    aspect: str = Field(max_length=300)


class ChecklistArgs(_Args):
    role: str = Field(min_length=2, max_length=200)
    topic: str = Field(min_length=2, max_length=300)
    regulations: list[str] = Field(max_length=5)


_ARG_MODELS: dict[str, type[_Args]] = {
    "retrieve": RetrieveArgs,
    "get_article": GetArticleArgs,
    "compare_articles": CompareArgs,
    "generate_checklist": ChecklistArgs,
}


# --------------------------------------------------------------------------- formatting


def _clean(text: str) -> str:
    # Source text must not be able to close or open its own delimiter.
    return _DELIMITER.sub("[source]", text)


def format_sources(items: list[EvidenceItem]) -> str:
    if not items:
        return "No matching provisions found."
    parts = []
    for it in items:
        score = f' score="{it.score:.3f}"' if it.score is not None else ""
        parts.append(
            f'<source id="{it.chunk_id}" label="{it.label}" kind="{it.kind}"'
            f' via="{it.origin}"{score}>\n{_clean(it.text)}\n</source>'
        )
    return "\n".join(parts)


# --------------------------------------------------------------------------- executor


class ToolExecutor:
    def __init__(self, retriever: Retriever, *, max_article_chunks: int = 25) -> None:
        self.retriever = retriever
        self.corpus: Corpus = retriever.searcher.corpus
        self.regulations = self.corpus.regulations
        self.max_article_chunks = max_article_chunks
        # The retriever's models are not safe to call from several request threads at once.
        self._lock = threading.Lock()

    # Tool specs sent to the model. strict=True: the API validates arguments against these.
    def specs(self) -> list[dict[str, Any]]:
        reg = {"type": "string", "enum": self.regulations}
        regs = {"type": "array", "items": reg, "description": "Empty list = all regulations."}
        art = {"type": "integer", "description": "Article number"}

        def tool(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": name,
                "description": description,
                "strict": True,
                "input_schema": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                    "additionalProperties": False,
                },
            }

        ref = {
            "type": "object",
            "properties": {"regulation": reg, "article": art},
            "required": ["regulation", "article"],
            "additionalProperties": False,
        }
        return [
            tool(
                "retrieve",
                "Search the corpus for the paragraphs most relevant to a query. Returns "
                "binding provisions (articles, annexes) ranked by relevance, plus related "
                "definitions and non-binding recitals as context.",
                {"query": {"type": "string"}, "regulations": regs},
            ),
            tool(
                "get_article",
                "Read an article in full, or one paragraph of it (paragraph 0 = whole article).",
                {"regulation": reg, "article": art, "paragraph": {"type": "integer"}},
            ),
            tool(
                "compare_articles",
                "Return two articles side by side so they can be compared on an aspect.",
                {"left": ref, "right": ref, "aspect": {"type": "string"}},
            ),
            tool(
                "generate_checklist",
                "Collect the obligation-bearing paragraphs ('shall ...') that apply to a role "
                "on a topic, per regulation, as material for a compliance checklist.",
                {"role": {"type": "string"}, "topic": {"type": "string"}, "regulations": regs},
            ),
        ]

    def execute(self, name: str, raw_args: dict[str, Any]) -> ToolOutcome:
        t0 = time.perf_counter()
        if name not in ALLOWED_TOOLS:
            available = ", ".join(ALLOWED_TOOLS)
            return ToolOutcome(
                f"Error: tool '{name}' does not exist. Available tools: {available}.",
                is_error=True,
                security_flag=f"disallowed_tool:{name}",
            )
        try:
            args = _ARG_MODELS[name].model_validate(raw_args)
            self._check_regulations(args)
        except (ValidationError, ValueError) as e:
            return ToolOutcome(f"Error: invalid arguments for {name}: {e}", is_error=True)

        handler: Callable[[Any], ToolOutcome] = getattr(self, f"_{name}")
        with self._lock:
            outcome = handler(args)
        outcome.elapsed_ms = (time.perf_counter() - t0) * 1000
        return outcome

    def _check_regulations(self, args: _Args) -> None:
        names: list[str] = []
        for value in args.model_dump().values():
            if isinstance(value, dict) and "regulation" in value:
                names.append(value["regulation"])
        if isinstance(args, GetArticleArgs):
            names.append(args.regulation)
        if isinstance(args, RetrieveArgs | ChecklistArgs):
            names += args.regulations
        unknown = sorted(set(names) - set(self.regulations))
        if unknown:
            raise ValueError(f"unknown regulation(s) {unknown}; choose from {self.regulations}")

    # --- tools

    def _retrieve(self, args: RetrieveArgs) -> ToolOutcome:
        result = self.retriever.retrieve(args.query, regulations=args.regulations or None)
        items = [_from_hit(h) for h in result.all_hits]
        return ToolOutcome(format_sources(items), evidence=items)

    def _get_article(self, args: GetArticleArgs) -> ToolOutcome:
        chunks = self.corpus.article_chunks(args.regulation, args.article, args.paragraph or None)[
            : self.max_article_chunks
        ]
        if not chunks:
            where = f"({args.paragraph})" if args.paragraph else ""
            return ToolOutcome(
                f"Error: {args.regulation} Art. {args.article}{where} is not in the corpus.",
                is_error=True,
            )
        items = [EvidenceItem.from_chunk(c, "get_article") for c in chunks]
        return ToolOutcome(format_sources(items), evidence=items)

    def _compare_articles(self, args: CompareArgs) -> ToolOutcome:
        sections, items = [], []
        for side, ref in (("LEFT", args.left), ("RIGHT", args.right)):
            chunks = self.corpus.article_chunks(ref.regulation, ref.article)
            if not chunks:
                return ToolOutcome(
                    f"Error: {ref.regulation} Art. {ref.article} is not in the corpus.",
                    is_error=True,
                )
            side_items = [
                EvidenceItem.from_chunk(c, "compare_articles")
                for c in chunks[: self.max_article_chunks]
            ]
            items += side_items
            sections.append(
                f"== {side}: {ref.regulation} Art. {ref.article} ==\n{format_sources(side_items)}"
            )
        header = f"Comparison aspect: {args.aspect}\n" if args.aspect else ""
        return ToolOutcome(header + "\n\n".join(sections), evidence=items)

    def _generate_checklist(self, args: ChecklistArgs) -> ToolOutcome:
        items: list[EvidenceItem] = []
        for reg in args.regulations or self.regulations:
            result = self.retriever.retrieve(
                f"obligations of {args.role}: {args.topic}", regulations=[reg], top_n=12
            )
            for h in result.hits:
                if h.chunk.kind is not ChunkKind.RECITAL and _OBLIGATION.search(h.chunk.text):
                    items.append(_from_hit(h, origin="generate_checklist"))
        intro = (
            f"Obligation-bearing provisions for '{args.role}' on '{args.topic}'. "
            "Check each one actually applies to this role before listing it.\n"
        )
        return ToolOutcome(intro + format_sources(items), evidence=items)


def _from_hit(h: Hit, origin: str | None = None) -> EvidenceItem:
    return EvidenceItem.from_chunk(h.chunk, origin or f"retrieve:{h.source}", h.rerank_score)
