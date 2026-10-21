"""The chunk data contract.

A ``ChunkRecord`` is the unit of retrieval *and* the unit of citation: every citation the
agent emits must resolve to exactly one record, and the verifier checks quotes against the
record's canonical ``text``. Anything that fails this contract is quarantined, never indexed.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

OFFICIAL_DOMAINS = frozenset({"eur-lex.europa.eu", "publications.europa.eu"})

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


class ChunkKind(StrEnum):
    ARTICLE = "article"
    RECITAL = "recital"
    ANNEX = "annex"


class TrustTier(StrEnum):
    OFFICIAL = "official"
    UNTRUSTED = "untrusted"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Cheap, model-independent token estimate (words + punctuation).

    Close enough to BPE counts for chunk sizing; exact counts are not part of the contract.
    """
    return len(_TOKEN_RE.findall(text))


def make_chunk_id(
    *,
    celex: str,
    kind: ChunkKind,
    version: str,
    article: int | None = None,
    paragraph: int | None = None,
    subparagraph: int | None = None,
    point: str | None = None,
    recital: int | None = None,
    annex: str | None = None,
    part: int | None = None,
) -> str:
    """Deterministic id, e.g. ``32016R0679:art5:p1:pta@oj-2016-05-04``."""
    parts = [celex]
    if kind is ChunkKind.ARTICLE:
        parts.append(f"art{article}")
        if paragraph is not None:
            parts.append(f"p{paragraph}")
        if subparagraph is not None:
            parts.append(f"sp{subparagraph}")
        if point is not None:
            parts.append(f"pt{point}")
    elif kind is ChunkKind.RECITAL:
        parts.append(f"rct{recital}")
    else:
        parts.append(f"anx{annex}")
    if part is not None:
        parts.append(f"part{part}")
    return ":".join(parts) + f"@{version}"


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    regulation: str = Field(pattern=r"^[A-Za-z0-9 ]{2,32}$")
    celex: str = Field(pattern=r"^[0-9]{5}[A-Z][0-9]{4}$")
    kind: ChunkKind

    article: int | None = Field(None, ge=1)
    paragraph: int | None = Field(None, ge=1)
    # Only set from the second subparagraph on; the first is implied by the paragraph.
    subparagraph: int | None = Field(None, ge=2)
    point: str | None = Field(None, pattern=r"^([a-z]{1,3}|[0-9]{1,3})$")
    recital: int | None = Field(None, ge=1)
    annex: str | None = Field(None, pattern=r"^[IVXLC]+$")
    part: int | None = Field(None, ge=1)

    heading_path: list[str] = Field(min_length=2)
    text: str = Field(min_length=20)
    source_url: HttpUrl
    version: str = Field(pattern=r"^[0-9A-Za-z.\-]+$")
    version_date: date
    language: Literal["en"] = "en"
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_tokens: int = Field(ge=1)
    trust_tier: TrustTier

    @model_validator(mode="after")
    def _structure_matches_kind(self) -> Self:
        located = {
            ChunkKind.ARTICLE: self.article is not None,
            ChunkKind.RECITAL: self.recital is not None,
            ChunkKind.ANNEX: self.annex is not None,
        }
        if not located[self.kind]:
            raise ValueError(f"{self.kind.value} chunk is missing its {self.kind.value} number")
        if sum(located.values()) != 1:
            raise ValueError("exactly one of article/recital/annex may be set")
        if self.kind is not ChunkKind.ARTICLE and (
            self.paragraph or self.subparagraph or self.point
        ):
            raise ValueError("paragraph/subparagraph/point are only valid on article chunks")
        return self

    @model_validator(mode="after")
    def _id_is_deterministic(self) -> Self:
        expected = make_chunk_id(
            celex=self.celex,
            kind=self.kind,
            version=self.version,
            article=self.article,
            paragraph=self.paragraph,
            subparagraph=self.subparagraph,
            point=self.point,
            recital=self.recital,
            annex=self.annex,
            part=self.part,
        )
        if self.chunk_id != expected:
            raise ValueError(f"chunk_id {self.chunk_id!r} != expected {expected!r}")
        return self

    @model_validator(mode="after")
    def _integrity(self) -> Self:
        if sha256_text(self.text) != self.content_sha256:
            raise ValueError("content_sha256 does not match text")
        if self.trust_tier is TrustTier.OFFICIAL and self.source_url.host not in OFFICIAL_DOMAINS:
            raise ValueError(f"official chunk from non-official host {self.source_url.host!r}")
        return self

    @property
    def citation_label(self) -> str:
        """Human-readable reference, e.g. ``GDPR Art. 5(1)(a)``."""
        if self.kind is ChunkKind.RECITAL:
            return f"{self.regulation} Recital {self.recital}"
        if self.kind is ChunkKind.ANNEX:
            return f"{self.regulation} Annex {self.annex}"
        label = f"{self.regulation} Art. {self.article}"
        if self.paragraph is not None:
            label += f"({self.paragraph})"
        if self.subparagraph is not None:
            label += f", subpara. {self.subparagraph}"
            if self.point is not None:
                label += f", point ({self.point})"
        elif self.point is not None:
            label += f"({self.point})"
        return label
