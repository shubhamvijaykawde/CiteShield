"""Retrieval plumbing with a deterministic fake encoder and in-memory Qdrant (no torch)."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from citeshield.config import IngestConfig
from citeshield.contracts.chunk import (
    ChunkKind,
)
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.validate import validate_document
from citeshield.retrieval.config import RecitalPolicy, RetrievalMode, RetrieverConfig
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.expand import Expander
from citeshield.retrieval.hybrid import Hit, HybridSearcher
from citeshield.retrieval.index import build_index
from citeshield.retrieval.pipeline import Retriever
from helpers import HashingEncoder, KeywordReranker, make_chunk


@pytest.fixture
def corpus(parsed: ParseResult) -> Corpus:
    return Corpus(validate_document(parsed.records, parsed.toc, IngestConfig()).valid)


@pytest.fixture
def searcher(corpus: Corpus) -> HybridSearcher:
    cfg = RetrieverConfig(collection="test")
    client = QdrantClient(":memory:")
    build_index(client, corpus, HashingEncoder(), cfg, batch_size=4)
    return HybridSearcher(client, corpus, HashingEncoder(), cfg)


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_each_mode_finds_the_obvious_chunk(searcher: HybridSearcher, mode: RetrievalMode) -> None:
    hits = searcher.search("notify the competent authority of every failed widget test", mode=mode)
    assert hits[0].chunk.point == "b"
    assert [h.retrieval_rank for h in hits] == list(range(1, len(hits) + 1))


def test_regulation_filter(searcher: HybridSearcher) -> None:
    assert searcher.search("widget", regulations=["Other Reg"]) == []
    assert searcher.search("widget", regulations=["Test Reg"])


def test_index_corpus_drift_is_refused(searcher: HybridSearcher, corpus: Corpus) -> None:
    searcher.corpus = Corpus(c for c in corpus if c.article != 1)
    with pytest.raises(LookupError, match="rebuild index"):
        searcher.search("This Regulation lays down rules on the testing of widgets")


def test_reranker_reorders_and_truncates(searcher: HybridSearcher) -> None:
    cfg = RetrieverConfig(collection="test", top_n=2)
    retriever = Retriever(searcher, KeywordReranker("research"), None, cfg)
    result = retriever.retrieve("widget records")
    assert len(result.hits) == 2
    assert "research" in result.hits[0].chunk.text
    assert result.top_score == 1.0
    assert set(result.timings_ms) == {"search", "rerank"}


@pytest.mark.parametrize("policy", list(RecitalPolicy))
def test_recital_policy(searcher: HybridSearcher, policy: RecitalPolicy) -> None:
    cfg = RetrieverConfig(collection="test", top_n=3, recital_policy=policy)
    retriever = Retriever(searcher, KeywordReranker("adequate records"), None, cfg)
    result = retriever.retrieve("Providers should keep adequate records of their test activities")
    hit_kinds = {h.chunk.kind for h in result.hits}
    if policy is RecitalPolicy.MIXED:
        assert result.hits[0].chunk.kind is ChunkKind.RECITAL
    else:
        assert ChunkKind.RECITAL not in hit_kinds
        assert [h.source for h in result.expansions] == ["recital", "recital"]
        # Evidence strength comes from binding text only, not from the recital's score.
        assert result.top_score is not None and result.top_score < 1.0


def test_definition_expansion_prefers_the_most_specific_term() -> None:
    defs = [
        make_chunk("AI Act", 3, "For the purposes of this Regulation:\n(1) ‘AI system’ means a "
                   "machine-based system.", point="1", heading="Article 3 — Definitions"),
        make_chunk("AI Act", 3, "For the purposes of this Regulation:\n(39) ‘emotion recognition "
                   "system’ means an AI system for inferring emotions.", point="39",
                   heading="Article 3 — Definitions"),
    ]  # fmt: skip
    hit_chunk = make_chunk("AI Act", 5, "1. The following AI practices shall be prohibited: ...",
                           paragraph=1)  # fmt: skip
    expander = Expander(Corpus([*defs, hit_chunk]))
    hits = [Hit(hit_chunk, 1, 1.0)]
    out = expander.expand(
        "Can I use an emotion recognition system at work?", hits, RetrieverConfig()
    )
    assert [h.chunk.point for h in out] == ["39"]
    assert out[0].source == "definition"
    # Definitions from regulations absent from the top hits are not pulled in.
    other = make_chunk("GDPR", 9, "1. Processing of special categories is prohibited.", paragraph=1)
    assert (
        expander.expand("emotion recognition system", [Hit(other, 1, 1.0)], RetrieverConfig()) == []
    )


def test_cross_reference_expansion() -> None:
    p1 = make_chunk("NIS2", 23, "1. Entities shall notify significant incidents.", paragraph=1)
    p4_text = (
        "4. For the purpose of notification under paragraph 1, the "
        "entities shall submit an early warning within 24 hours."
    )
    p4 = make_chunk("NIS2", 23, p4_text, paragraph=4)
    expander = Expander(Corpus([p1, p4]))
    out = expander.expand("incident deadlines", [Hit(p4, 1, 1.0)], RetrieverConfig())
    assert [(h.chunk.paragraph, h.source) for h in out] == [(1, "cross_ref")]
    assert expander.expand("x", [Hit(p4, 1, 1.0)], RetrieverConfig(max_expansions=0)) == []
