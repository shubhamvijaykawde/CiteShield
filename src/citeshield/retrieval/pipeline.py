"""The retriever the agent's tools call: search -> rerank -> expand, with timings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from citeshield.config import Settings, config_hash
from citeshield.contracts.chunk import ChunkKind, TrustTier, corpus_hash
from citeshield.retrieval.config import RecitalPolicy, RetrievalMode, RetrieverConfig
from citeshield.retrieval.corpus import Corpus, embedding_text
from citeshield.retrieval.embed import BGEM3Encoder
from citeshield.retrieval.expand import Expander
from citeshield.retrieval.hybrid import Hit, HybridSearcher
from citeshield.retrieval.index import open_client, read_manifest
from citeshield.retrieval.rerank import CrossEncoderReranker, Reranker


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    expansions: list[Hit] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    config_hash: str = ""

    @property
    def top_score(self) -> float | None:
        """Best rerank score; the refusal policy uses it as an evidence-strength signal."""
        scores = [h.rerank_score for h in self.hits if h.rerank_score is not None]
        return max(scores) if scores else None

    @property
    def all_hits(self) -> list[Hit]:
        return [*self.hits, *self.expansions]


class Retriever:
    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker | None,
        expander: Expander | None,
        cfg: RetrieverConfig,
    ) -> None:
        self.searcher = searcher
        self.reranker = reranker
        self.expander = expander
        self.cfg = cfg
        self.config_hash = config_hash(cfg)

    @classmethod
    def from_settings(cls, settings: Settings, cfg: RetrieverConfig | None = None) -> Retriever:
        cfg = cfg or settings.retriever
        corpus = Corpus.load(settings.chunks_path)
        manifest = read_manifest(settings.index_dir)
        if manifest.corpus_hash != corpus_hash(corpus):
            raise RuntimeError(
                f"index was built for corpus {manifest.corpus_hash}, but chunks.jsonl is "
                f"{corpus_hash(corpus)}; run citeshield-index"
            )
        if manifest.embed_model != cfg.embed_model:
            raise RuntimeError(f"index embedded with {manifest.embed_model}, not {cfg.embed_model}")
        encoder = BGEM3Encoder(
            cfg.embed_model,
            device=cfg.effective_embed_device,
            dtype=cfg.embed_dtype,
            max_length=cfg.max_length,
            batch_size=cfg.embed_batch_size,
        )
        reranker = (
            CrossEncoderReranker(
                cfg.rerank_model,
                device=cfg.effective_rerank_device,
                max_length=cfg.max_length,
                batch_size=cfg.rerank_batch_size,
            )
            if cfg.use_reranker
            else None
        )
        client = open_client(settings.index_dir, settings.qdrant_url)
        return cls(HybridSearcher(client, corpus, encoder, cfg), reranker, Expander(corpus), cfg)

    def retrieve(
        self,
        query: str,
        *,
        regulations: list[str] | None = None,
        trust_tiers: list[TrustTier] | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool | None = None,
        top_n: int | None = None,
        recital_policy: RecitalPolicy | None = None,
    ) -> RetrievalResult:
        top_n = top_n or self.cfg.top_n
        recital_policy = recital_policy or self.cfg.recital_policy
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        candidates = self.searcher.search(
            query, regulations=regulations, trust_tiers=trust_tiers, mode=mode
        )
        timings["search"] = (time.perf_counter() - t0) * 1000

        do_rerank = self.reranker is not None and (rerank if rerank is not None else True)
        if do_rerank and candidates:
            assert self.reranker is not None
            t0 = time.perf_counter()
            scores = self.reranker.score(query, [embedding_text(h.chunk) for h in candidates])
            for hit, score in zip(candidates, scores, strict=True):
                hit.rerank_score = score
            candidates.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
            timings["rerank"] = (time.perf_counter() - t0) * 1000

        recital_context: list[Hit] = []
        if recital_policy is RecitalPolicy.CONTEXT:
            operative = [h for h in candidates if h.chunk.kind is not ChunkKind.RECITAL]
            recitals = [h for h in candidates if h.chunk.kind is ChunkKind.RECITAL]
            candidates = operative
            for h in recitals[: self.cfg.max_recital_context]:
                h.source = "recital"
                recital_context.append(h)

        hits = candidates[:top_n]
        expansions: list[Hit] = []
        if self.expander is not None:
            t0 = time.perf_counter()
            expansions = self.expander.expand(query, hits, self.cfg)
            timings["expand"] = (time.perf_counter() - t0) * 1000
        expansions += recital_context

        return RetrievalResult(query, hits, expansions, timings, self.config_hash)
