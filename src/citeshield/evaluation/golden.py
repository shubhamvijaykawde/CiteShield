"""Golden question set and paragraph-level citation matching.

Expected citations are written the way lawyers write them and parsed here:

    "GDPR 6(1)"            article 6, paragraph 1 (any point/subparagraph)
    "AI Act 5(1)(h)"       article 5, paragraph 1, point (h)
    "GDPR 4(5)"            article 4 has no numbered paragraphs, so (5) is definition point 5
    "AI Act 113"           any part of article 113
    "AI Act Annex III.4"   item 4 of Annex III
    "GDPR Recital 71"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from citeshield.contracts.chunk import ChunkKind, ChunkRecord

_ARTICLE_REF = re.compile(
    r"^(?P<reg>.+?) (?P<art>\d+)(?:\((?P<para>\d+)\))?(?:\((?P<pt>[a-z]{1,3}|\d{1,3})\))?$"
)
_ANNEX_REF = re.compile(r"^(?P<reg>.+?) Annex (?P<annex>[IVXLC]+)(?:\.(?P<item>\d+))?$")
_RECITAL_REF = re.compile(r"^(?P<reg>.+?) Recital (?P<rct>\d+)$")


@dataclass(frozen=True)
class Ref:
    regulation: str
    kind: ChunkKind
    article: int | None = None
    paragraph: int | None = None
    point: str | None = None
    annex: str | None = None
    item: int | None = None
    recital: int | None = None
    label: str = ""

    def matches(self, c: ChunkRecord) -> bool:
        if c.regulation != self.regulation or c.kind is not self.kind:
            return False
        if self.kind is ChunkKind.RECITAL:
            return c.recital == self.recital
        if self.kind is ChunkKind.ANNEX:
            if c.annex != self.annex:
                return False
            return self.item is None or re.match(rf"^{self.item}\.\s", c.text) is not None
        if c.article != self.article:
            return False
        if self.paragraph is not None:
            # Articles without numbered paragraphs (definitions) number their points instead.
            in_paragraph = c.paragraph == self.paragraph
            as_point = c.paragraph is None and c.point == str(self.paragraph)
            if not (in_paragraph or as_point):
                return False
        return self.point is None or c.point == self.point


def parse_ref(text: str) -> Ref:
    if m := _RECITAL_REF.match(text):
        return Ref(m["reg"], ChunkKind.RECITAL, recital=int(m["rct"]), label=text)
    if m := _ANNEX_REF.match(text):
        item = int(m["item"]) if m["item"] else None
        return Ref(m["reg"], ChunkKind.ANNEX, annex=m["annex"], item=item, label=text)
    if m := _ARTICLE_REF.match(text):
        return Ref(
            m["reg"],
            ChunkKind.ARTICLE,
            article=int(m["art"]),
            paragraph=int(m["para"]) if m["para"] else None,
            point=m["pt"],
            label=text,
        )
    raise ValueError(f"cannot parse citation {text!r}")


class Category(StrEnum):
    LOOKUP = "lookup"
    DEFINITION = "definition"
    MULTI_PARAGRAPH = "multi_paragraph"
    CROSS_REGULATION = "cross_regulation"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    CHECKLIST = "checklist"
    UNANSWERABLE = "unanswerable"


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    question: str
    category: Category
    expected: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    notes: str | None = None
    # Short gold answer for the judge and the reference-based metrics; None for refusals.
    reference_answer: str | None = None
    # Items drafted by the assistant start unreviewed until a human has checked the citations.
    reviewed: bool = True

    @field_validator("expected")
    @classmethod
    def _parseable(cls, v: list[str]) -> list[str]:
        for ref in v:
            parse_ref(ref)
        return v

    @property
    def refs(self) -> list[Ref]:
        return [parse_ref(r) for r in self.expected]


def load_golden(path: Path) -> list[GoldenItem]:
    with path.open(encoding="utf-8") as f:
        items = [GoldenItem.model_validate_json(line) for line in f if line.strip()]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate golden ids")
    for i in items:
        if i.should_refuse == bool(i.expected):
            raise ValueError(f"{i.id}: refusal items have no citations, others need some")
        if i.should_refuse and i.reference_answer:
            raise ValueError(f"{i.id}: refusal items have no reference answer")
    return items


def unresolved_refs(items: list[GoldenItem], chunks: list[ChunkRecord]) -> list[str]:
    """Golden refs that match no chunk at all: typos or wrong paragraph numbers."""
    return [
        f"{item.id}: {ref.label}"
        for item in items
        for ref in item.refs
        if not any(ref.matches(c) for c in chunks)
    ]
