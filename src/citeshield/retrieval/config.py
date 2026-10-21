"""Retriever configuration. Its ``config_hash`` is logged with every eval run and trace."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class RecitalPolicy(StrEnum):
    # Recitals compete with articles for the top-n slots.
    MIXED = "mixed"
    # Only operative text (articles, annexes) is ranked; the best recitals are attached
    # separately as non-binding interpretive context. Measured in eval/results: the reranker
    # otherwise prefers recitals' explanatory prose over the binding provision.
    CONTEXT = "context"


class RetrieverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    collection: str = "eu_regs"

    mode: RetrievalMode = RetrievalMode.HYBRID
    dense_k: int = Field(40, ge=1)
    sparse_k: int = Field(40, ge=1)
    fused_k: int = Field(40, ge=1, description="Candidates passed to the reranker.")
    use_reranker: bool = True
    top_n: int = Field(8, ge=1, description="Hits returned after reranking.")
    recital_policy: RecitalPolicy = RecitalPolicy.CONTEXT
    max_recital_context: int = Field(2, ge=0)

    expand_definitions: bool = True
    expand_cross_refs: bool = True
    max_expansions: int = Field(4, ge=0)

    max_length: int = Field(512, ge=32, description="Token limit for encoder/reranker inputs.")
    embed_batch_size: int = Field(16, ge=1)
    rerank_batch_size: int = Field(16, ge=1)
    device: str | None = Field(None, description="None = cuda if available, else cpu.")
