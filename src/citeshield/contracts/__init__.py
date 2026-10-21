"""Data contracts shared by ingestion, retrieval, the agent and evaluation."""

from citeshield.contracts.answer import (
    AgentAnswer,
    AnswerStatus,
    ChecklistItem,
    Citation,
    Claim,
    RefusalReason,
)
from citeshield.contracts.chunk import ChunkKind, ChunkRecord, TrustTier, make_chunk_id

__all__ = [
    "AgentAnswer",
    "AnswerStatus",
    "ChecklistItem",
    "ChunkKind",
    "ChunkRecord",
    "Citation",
    "Claim",
    "RefusalReason",
    "TrustTier",
    "make_chunk_id",
]
