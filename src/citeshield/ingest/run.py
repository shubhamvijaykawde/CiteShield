"""Ingestion entry point: ``uv run citeshield-ingest [--only GDPR --only "AI Act"]``.

Writes ``data/processed/chunks.jsonl`` only when every document passes its gates, so a
partially broken corpus can never reach the index. The report is always written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from citeshield.config import Settings, config_hash
from citeshield.contracts.chunk import ChunkRecord, corpus_hash
from citeshield.ingest.fetch import fetch_xhtml
from citeshield.ingest.manifest import load_manifest
from citeshield.ingest.parse_xhtml import parse_document
from citeshield.ingest.quarantine import reason_counts, write_quarantine
from citeshield.ingest.validate import validate_document


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", action="append", help="Regulation name(s) to ingest.")
    ap.add_argument("--refresh", action="store_true", help="Re-download even if cached.")
    args = ap.parse_args(argv)

    settings = Settings()
    cfg = settings.ingest
    manifest = load_manifest(settings.manifest_path)
    sources = [
        s for s in manifest.sources if s.enabled and (not args.only or s.regulation in args.only)
    ]
    if not sources:
        print(f"no enabled sources match {args.only}", file=sys.stderr)
        return 2

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine_path = settings.quarantine_dir / f"{run_id}.jsonl"
    report: dict[str, Any] = {"run_id": run_id, "ingest_config_hash": config_hash(cfg)}
    per_source: dict[str, Any] = {}
    corpus: list[ChunkRecord] = []
    failed = False

    for src in sources:
        doc = fetch_xhtml(src.celex, settings.raw_dir, refresh=args.refresh)
        parsed = parse_document(doc.path.read_bytes(), src, max_tokens=cfg.max_chunk_tokens)
        result = validate_document(parsed.records, parsed.toc, cfg)
        write_quarantine(result.quarantined, quarantine_path, run_id=run_id)

        source_failed = bool(result.doc_errors) or result.quarantine_rate > cfg.max_quarantine_rate
        failed |= source_failed
        corpus.extend(result.valid)
        per_source[src.regulation] = {
            "celex": src.celex,
            "version": src.version,
            "raw_sha256": doc.sha256,
            "resolved_url": doc.resolved_url,
            "parsed": len(parsed.records),
            "valid": len(result.valid),
            "quarantined": len(result.quarantined),
            "quarantine_rate": round(result.quarantine_rate, 4),
            "by_kind": dict(Counter(r.kind.value for r in result.valid)),
            "toc": {
                "articles": len(parsed.toc.articles),
                "recitals": len(parsed.toc.recitals),
                "annexes": len(parsed.toc.annexes),
            },
            "quarantine_reasons": reason_counts(result.quarantined),
            "doc_errors": result.doc_errors,
            "parser_warnings": parsed.warnings,
            "passed": not source_failed,
        }

    report["sources"] = per_source
    report["total_valid"] = len(corpus)
    report["total_quarantined"] = sum(s["quarantined"] for s in per_source.values())
    report["corpus_hash"] = corpus_hash(corpus) if not failed else None
    report["passed"] = not failed

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    (settings.processed_dir / "ingest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not failed:
        with (settings.processed_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in corpus:
                f.write(chunk.model_dump_json() + "\n")

    _print_summary(report)
    return 1 if failed else 0


def _print_summary(report: dict[str, Any]) -> None:
    print(f"{'regulation':<10} {'parsed':>7} {'valid':>7} {'quar.':>6} {'rate':>7}  status")
    for name, s in report["sources"].items():
        status = "ok" if s["passed"] else "FAIL"
        print(
            f"{name:<10} {s['parsed']:>7} {s['valid']:>7} {s['quarantined']:>6} "
            f"{s['quarantine_rate']:>7.2%}  {status}"
        )
        for err in s["doc_errors"]:
            print(f"    doc error: {err}")
        for reason, n in list(s["quarantine_reasons"].items())[:5]:
            print(f"    {n:>4} x {reason}")
    print(f"corpus_hash={report['corpus_hash']} passed={report['passed']}")


if __name__ == "__main__":
    raise SystemExit(main())
