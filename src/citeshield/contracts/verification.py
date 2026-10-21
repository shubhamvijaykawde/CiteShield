"""The verification report: what was checked for every claim, citation and sentence, and why.

It travels next to the ``AgentAnswer`` (on ``AgentRun`` and in the final SSE payload); the
answer contract itself is unchanged apart from two verifier-only refusal reasons.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CitationStatus(StrEnum):
    VALID = "valid"
    FABRICATED = "fabricated"  # id not in the canonical corpus
    VERSION_MISMATCH = "version_mismatch"  # provision exists, but at another @version
    NOT_RETRIEVED = "not_retrieved"  # in the corpus, but no tool returned it this run
    NON_CITABLE = "non_citable"  # e.g. trust_tier=untrusted


class QuoteMatch(StrEnum):
    MATCHED = "matched"  # normalised quote is a substring of the canonical text
    NOT_FOUND = "not_found"
    NOT_CHECKED = "not_checked"  # citation invalid, nothing to compare against


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"  # sources do not entail it
    CONTRADICTED = "contradicted"  # a cited source contradicts it
    CONFLICTING = "conflicting"  # one cited source entails it, another contradicts it
    CITATION_INVALID = "citation_invalid"  # no citation passed the id and quote checks


class Entailment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entailment: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)
    contradiction: float = Field(ge=0, le=1)


class CitationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    label: str | None = None
    status: CitationStatus
    quote_match: QuoteMatch
    quote_similarity: float | None = Field(None, description="Diagnostic only; never accepted.")
    entailment: Entailment | None = None
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.status is CitationStatus.VALID and self.quote_match is QuoteMatch.MATCHED


class ClaimCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["claim", "checklist"]
    index: int
    text: str
    verdict: ClaimVerdict
    citations: list[CitationCheck]
    combined_entailment: Entailment | None = None
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SentenceCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["summary", "caveat"]
    text: str
    kept: bool
    entailment: float | None = None
    reason: str | None = None


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimCheck] = Field(default_factory=list)
    sentences: list[SentenceCheck] = Field(default_factory=list)
    retrieval_top_score: float | None = None
    nli_model: str
    thresholds: dict[str, float]

    @property
    def supported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict is ClaimVerdict.SUPPORTED]

    @property
    def failed(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.verdict is not ClaimVerdict.SUPPORTED]

    @property
    def support_ratio(self) -> float | None:
        return len(self.supported) / len(self.claims) if self.claims else None

    @property
    def all_claims_supported(self) -> bool:
        return bool(self.claims) and not self.failed
