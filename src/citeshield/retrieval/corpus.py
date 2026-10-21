"""In-memory view of the validated corpus (chunks.jsonl), keyed by chunk_id.

This is the canonical text the verifier checks quotes against; the vector index only stores
ids and filterable metadata.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from citeshield.contracts.chunk import ChunkKind, ChunkRecord


class Corpus:
    def __init__(self, chunks: Iterable[ChunkRecord]) -> None:
        self._by_id: dict[str, ChunkRecord] = {}
        for c in chunks:
            if c.chunk_id in self._by_id:
                raise ValueError(f"duplicate chunk_id {c.chunk_id}")
            self._by_id[c.chunk_id] = c

    @classmethod
    def load(cls, path: Path) -> Corpus:
        with path.open(encoding="utf-8") as f:
            return cls(ChunkRecord.model_validate_json(line) for line in f if line.strip())

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[ChunkRecord]:
        return iter(self._by_id.values())

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    def get(self, chunk_id: str) -> ChunkRecord | None:
        return self._by_id.get(chunk_id)

    def __getitem__(self, chunk_id: str) -> ChunkRecord:
        return self._by_id[chunk_id]

    def article_chunks(
        self, regulation: str, article: int, paragraph: int | None = None
    ) -> list[ChunkRecord]:
        return [
            c
            for c in self._by_id.values()
            if c.kind is ChunkKind.ARTICLE
            and c.regulation == regulation
            and c.article == article
            and (paragraph is None or c.paragraph == paragraph)
        ]

    @property
    def regulations(self) -> list[str]:
        return sorted({c.regulation for c in self._by_id.values()})


def embedding_text(chunk: ChunkRecord) -> str:
    """Text that gets embedded: the breadcrumb, then the body.

    The breadcrumb carries the article title ("Article 22 — Automated individual
    decision-making"), which is often the strongest retrieval signal.
    """
    return " › ".join([chunk.regulation, *chunk.heading_path[1:]]) + "\n" + chunk.text
