"""First-stage search: dense, sparse, or both fused with Reciprocal Rank Fusion in Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient, models

from citeshield.contracts.chunk import ChunkRecord, TrustTier
from citeshield.retrieval.config import RetrievalMode, RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.embed import EncodedText, Encoder
from citeshield.retrieval.index import DENSE, SPARSE

HitSource = Literal["search", "definition", "cross_ref", "recital"]


@dataclass
class Hit:
    chunk: ChunkRecord
    retrieval_rank: int
    retrieval_score: float
    rerank_score: float | None = None
    source: HitSource = "search"


def build_filter(
    regulations: list[str] | None, trust_tiers: list[TrustTier] | None
) -> models.Filter | None:
    must: list[models.Condition] = []
    if regulations:
        must.append(
            models.FieldCondition(key="regulation", match=models.MatchAny(any=list(regulations)))
        )
    if trust_tiers:
        must.append(
            models.FieldCondition(
                key="trust_tier", match=models.MatchAny(any=[t.value for t in trust_tiers])
            )
        )
    return models.Filter(must=must) if must else None


class HybridSearcher:
    def __init__(
        self, client: QdrantClient, corpus: Corpus, encoder: Encoder, cfg: RetrieverConfig
    ) -> None:
        self.client = client
        self.corpus = corpus
        self.encoder = encoder
        self.cfg = cfg

    def search(
        self,
        query: str,
        *,
        regulations: list[str] | None = None,
        trust_tiers: list[TrustTier] | None = None,
        mode: RetrievalMode | None = None,
        limit: int | None = None,
    ) -> list[Hit]:
        mode = mode or self.cfg.mode
        limit = limit or self.cfg.fused_k
        enc = self.encoder.encode([query])[0]
        flt = build_filter(regulations, trust_tiers)
        sparse = _sparse_vector(enc)

        if mode is RetrievalMode.DENSE or (mode is RetrievalMode.HYBRID and sparse is None):
            res = self.client.query_points(
                self.cfg.collection, query=enc.dense, using=DENSE, query_filter=flt, limit=limit
            )
        elif mode is RetrievalMode.SPARSE:
            if sparse is None:
                return []
            res = self.client.query_points(
                self.cfg.collection, query=sparse, using=SPARSE, query_filter=flt, limit=limit
            )
        else:
            res = self.client.query_points(
                self.cfg.collection,
                prefetch=[
                    models.Prefetch(
                        query=enc.dense, using=DENSE, filter=flt, limit=self.cfg.dense_k
                    ),
                    models.Prefetch(
                        query=sparse, using=SPARSE, filter=flt, limit=self.cfg.sparse_k
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )

        hits: list[Hit] = []
        for point in res.points:
            chunk_id = (point.payload or {}).get("chunk_id")
            chunk = self.corpus.get(str(chunk_id))
            if chunk is None:
                # Index and corpus drifted apart; never serve text that is not in the corpus.
                raise LookupError(f"indexed chunk {chunk_id!r} is not in the corpus; rebuild index")
            hits.append(Hit(chunk=chunk, retrieval_rank=len(hits) + 1, retrieval_score=point.score))
        return hits


def _sparse_vector(enc: EncodedText) -> models.SparseVector | None:
    if not enc.sparse:
        return None
    return models.SparseVector(indices=list(enc.sparse.keys()), values=list(enc.sparse.values()))
