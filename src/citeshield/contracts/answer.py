"""The agent's output contract.

The agent can only answer by producing an ``AgentAnswer``. Every claim carries at least one
paragraph-level citation with a verbatim quote, and a refusal must say why. The verifier and
the evaluation harness consume exactly this object.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    REFUSED = "refused"


class RefusalReason(StrEnum):
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"
    LEGAL_ADVICE_REQUEST = "legal_advice_request"
    CONFLICTING_SOURCES = "conflicting_sources"
    INJECTION_DETECTED = "injection_detected"
    # Set only by the verifier (milestone 5), never offered to the drafting model.
    INVALID_CITATIONS = "invalid_citations"
    UNSUPPORTED_CLAIMS = "unsupported_claims"


QUOTE_MIN_CHARS, QUOTE_MAX_CHARS = 8, 400

VERIFIER_ONLY_REASONS = frozenset(
    {RefusalReason.INVALID_CITATIONS, RefusalReason.UNSUPPORTED_CLAIMS}
)


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(description="Must be the id of a chunk returned by a tool in this run.")
    quote: str = Field(
        min_length=QUOTE_MIN_CHARS,
        max_length=QUOTE_MAX_CHARS,
        description="Verbatim span from the cited chunk that supports the claim.",
    )


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
    citations: list[Citation] = Field(min_length=1)


class ChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation: str = Field(min_length=1, max_length=500)
    applies_to: str = Field(min_length=1, max_length=200)
    citations: list[Citation] = Field(min_length=1)


class AgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    summary: str = Field(max_length=2000)
    claims: list[Claim] = Field(default_factory=list)
    checklist: list[ChecklistItem] | None = None
    refusal_reason: RefusalReason | None = None
    caveats: list[str] = Field(default_factory=list)
    corpus_version: str | None = None

    @model_validator(mode="after")
    def _status_consistency(self) -> Self:
        if self.status is AnswerStatus.REFUSED:
            if self.refusal_reason is None:
                raise ValueError("a refused answer must give a refusal_reason")
            if self.claims or self.checklist:
                raise ValueError("a refused answer must not contain claims or a checklist")
        else:
            if not self.claims and not self.checklist:
                raise ValueError(
                    "an answer must contain at least one cited claim or checklist item"
                )
            if self.status is AnswerStatus.ANSWERED and self.refusal_reason is not None:
                raise ValueError("refusal_reason is only allowed on refused or partial answers")
        return self

    def all_citations(self) -> list[Citation]:
        cites = [c for claim in self.claims for c in claim.citations]
        cites += [c for item in self.checklist or [] for c in item.citations]
        return cites
