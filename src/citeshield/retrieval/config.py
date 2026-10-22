"""Retriever configuration. Its ``config_hash`` is logged with every eval run and trace."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

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
    # Per-model overrides of ``device`` (None = use ``device``). The answer eval runs both on
    # CPU so the 8 GB GPU is left to the Ollama model (citeshield-eval --embed-device).
    # Left unset they are dropped from ``config_hash`` (exclude_none), so hashes don't move.
    embed_device: str | None = None
    rerank_device: str | None = None
    # Opt-in embedder weight dtype; None = fp16 on cuda, fp32 on cpu (the encoder's rule).
    # "bfloat16" on cpu saves ~1.1 GB for the answer eval at ~4% retrieval latency on the dev
    # box (no native BF16), with identical rankings on the fixed benchmark. Unset it is
    # dropped from config_hash, like the device overrides.
    embed_dtype: Literal["float32", "bfloat16"] | None = None

    @property
    def effective_embed_device(self) -> str | None:
        return self.embed_device or self.device

    @property
    def effective_rerank_device(self) -> str | None:
        return self.rerank_device or self.device
