"""Build and open the Qdrant collection (named dense + sparse vectors per chunk).

Only ids and filterable metadata go into Qdrant payloads; text is resolved from the
validated corpus so the index can never serve text that bypassed the data contract.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

from citeshield.config import config_hash
from citeshield.contracts.chunk import ChunkRecord, corpus_hash
from citeshield.retrieval.config import RetrieverConfig
from citeshield.retrieval.corpus import Corpus, embedding_text
from citeshield.retrieval.embed import Encoder

DENSE = "dense"
SPARSE = "sparse"
_ID_NAMESPACE = uuid.UUID("6f1c3a52-6a0e-4c1e-9d2b-3c5e1e0b7a11")


@dataclass(frozen=True)
class IndexManifest:
    collection: str
    corpus_hash: str
    embed_model: str
    n_points: int
    retriever_config_hash: str
    built_at: str


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def open_client(index_dir: Path, url: str | None = None) -> QdrantClient:
    if url:
        return QdrantClient(url=url)
    index_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(index_dir / "qdrant"))


def payload(chunk: ChunkRecord) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "regulation": chunk.regulation,
        "kind": chunk.kind.value,
        "article": chunk.article,
        "paragraph": chunk.paragraph,
        "trust_tier": chunk.trust_tier.value,
    }


def build_index(
    client: QdrantClient,
    corpus: Corpus,
    encoder: Encoder,
    cfg: RetrieverConfig,
    *,
    batch_size: int = 64,
    progress: Callable[[int, int], None] | None = None,
) -> IndexManifest:
    if client.collection_exists(cfg.collection):
        client.delete_collection(cfg.collection)
    client.create_collection(
        cfg.collection,
        vectors_config={
            DENSE: models.VectorParams(size=encoder.dim, distance=models.Distance.COSINE)
        },
        # bge-m3 lexical weights are already learned term importances: plain dot product, no IDF.
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )

    chunks = list(corpus)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        encoded = encoder.encode([embedding_text(c) for c in batch])
        client.upsert(
            cfg.collection,
            points=[
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector={
                        DENSE: e.dense,
                        SPARSE: models.SparseVector(
                            indices=list(e.sparse.keys()), values=list(e.sparse.values())
                        ),
                    },
                    payload=payload(c),
                )
                for c, e in zip(batch, encoded, strict=True)
            ],
        )
        if progress:
            progress(min(start + batch_size, len(chunks)), len(chunks))

    return IndexManifest(
        collection=cfg.collection,
        corpus_hash=corpus_hash(chunks),
        embed_model=cfg.embed_model,
        n_points=client.count(cfg.collection).count,
        retriever_config_hash=config_hash(cfg),
        built_at=datetime.now(UTC).isoformat(),
    )


def write_manifest(manifest: IndexManifest, index_dir: Path) -> None:
    (index_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), "utf-8")


def read_manifest(index_dir: Path) -> IndexManifest:
    return IndexManifest(**json.loads((index_dir / "manifest.json").read_text("utf-8")))
