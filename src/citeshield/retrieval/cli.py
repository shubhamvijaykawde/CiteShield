"""``citeshield-index`` (build the index) and ``citeshield-search`` (ad-hoc queries)."""

from __future__ import annotations

import argparse
import sys
import time

from citeshield.config import Settings
from citeshield.retrieval.config import RetrievalMode
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.embed import BGEM3Encoder
from citeshield.retrieval.index import build_index, open_client, write_manifest
from citeshield.retrieval.pipeline import Retriever


def index_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Embed chunks.jsonl into Qdrant.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    settings = Settings()
    cfg = settings.retriever
    corpus = Corpus.load(settings.chunks_path)
    encoder = BGEM3Encoder(
        cfg.embed_model,
        device=args.device or cfg.device,
        max_length=cfg.max_length,
        batch_size=cfg.embed_batch_size,
    )
    print(f"embedding {len(corpus)} chunks with {cfg.embed_model} on {encoder.device}")
    client = open_client(settings.index_dir, settings.qdrant_url)
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total}  {time.perf_counter() - t0:6.1f}s", end="", file=sys.stderr)

    manifest = build_index(client, corpus, encoder, cfg, progress=progress)
    print(file=sys.stderr)
    write_manifest(manifest, settings.index_dir)
    print(f"indexed {manifest.n_points} points, corpus {manifest.corpus_hash}")
    return 0


def search_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Query the index.")
    ap.add_argument("query")
    ap.add_argument("--regulation", action="append")
    ap.add_argument("--mode", choices=[m.value for m in RetrievalMode], default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("-n", type=int, default=None)
    args = ap.parse_args(argv)

    retriever = Retriever.from_settings(Settings())
    result = retriever.retrieve(
        args.query,
        regulations=args.regulation,
        mode=RetrievalMode(args.mode) if args.mode else None,
        rerank=not args.no_rerank,
        top_n=args.n,
    )
    for h in result.all_hits:
        score = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "  -  "
        print(f"{score}  #{h.retrieval_rank:<3} {h.source:<10} {h.chunk.citation_label}")
        print(f"         {h.chunk.text[:140].replace(chr(10), ' ')}")
    timings = ", ".join(f"{k}={v:.0f}ms" for k, v in result.timings_ms.items())
    print(f"[{timings}] config={result.config_hash}")
    return 0
