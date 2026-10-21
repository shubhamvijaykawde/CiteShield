"""Citation validity against the canonical corpus and the current run's evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container
from dataclasses import dataclass

from citeshield.contracts.chunk import ChunkRecord, TrustTier
from citeshield.contracts.verification import CitationStatus
from citeshield.retrieval.corpus import Corpus


@dataclass(frozen=True)
class CitationResult:
    status: CitationStatus
    chunk: ChunkRecord | None
    reason: str | None = None


class CitationChecker:
    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus
        self._versions: dict[str, list[str]] = defaultdict(list)
        for c in corpus:
            self._versions[c.chunk_id.split("@", 1)[0]].append(c.chunk_id)

    def check(self, chunk_id: str, retrieved: Container[str]) -> CitationResult:
        chunk = self.corpus.get(chunk_id)
        if chunk is None:
            base = chunk_id.split("@", 1)[0]
            if base in self._versions:
                known = ", ".join(self._versions[base])
                return CitationResult(
                    CitationStatus.VERSION_MISMATCH,
                    None,
                    f"{chunk_id!r} is not in the corpus; the corpus has {known}",
                )
            return CitationResult(
                CitationStatus.FABRICATED, None, f"{chunk_id!r} does not exist in the corpus"
            )
        if chunk.trust_tier is not TrustTier.OFFICIAL:
            return CitationResult(
                CitationStatus.NON_CITABLE,
                chunk,
                f"{chunk_id!r} is a {chunk.trust_tier.value} source and cannot be cited",
            )
        if chunk_id not in retrieved:
            return CitationResult(
                CitationStatus.NOT_RETRIEVED,
                chunk,
                f"{chunk_id!r} was not returned by any tool in this run",
            )
        return CitationResult(CitationStatus.VALID, chunk)
