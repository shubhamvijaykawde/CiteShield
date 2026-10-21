"""Retrieval evaluation: paragraph-level Recall@k / Hit@k / MRR across retriever variants.

    uv run citeshield-eval-retrieval                 # all variants
    uv run citeshield-eval-retrieval --variants hybrid+rerank --show-misses

Answerable questions feed the ranking metrics. Unanswerable ones are kept to compare top
rerank scores, which is the first signal for calibrating the refusal threshold (milestone 5).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from citeshield.config import PROJECT_ROOT, Settings, config_hash
from citeshield.contracts.chunk import ChunkRecord, corpus_hash
from citeshield.evaluation.golden import GoldenItem, Ref, load_golden, unresolved_refs
from citeshield.retrieval.config import RecitalPolicy, RetrievalMode
from citeshield.retrieval.pipeline import RetrievalResult, Retriever

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
KS = (1, 3, 5, 8, 20)
DEPTH = max(KS)


@dataclass(frozen=True)
class Variant:
    name: str
    mode: RetrievalMode
    rerank: bool
    recitals: RecitalPolicy = RecitalPolicy.MIXED


VARIANTS = {
    v.name: v
    for v in (
        Variant("dense", RetrievalMode.DENSE, rerank=False),
        Variant("sparse", RetrievalMode.SPARSE, rerank=False),
        Variant("hybrid", RetrievalMode.HYBRID, rerank=False),
        Variant("hybrid+rerank", RetrievalMode.HYBRID, rerank=True),
        Variant("hybrid+ctx", RetrievalMode.HYBRID, rerank=False, recitals=RecitalPolicy.CONTEXT),
        Variant(
            "hybrid+rerank+ctx", RetrievalMode.HYBRID, rerank=True, recitals=RecitalPolicy.CONTEXT
        ),
    )
}


@dataclass
class QuestionResult:
    id: str
    category: str
    ranked: list[str]
    first_relevant_rank: int | None
    refs_found_at: dict[str, int | None]
    expansion_ids: list[str] = field(default_factory=list)
    expansion_recovered: list[str] = field(default_factory=list)
    top_score: float | None = None


def score_question(item: GoldenItem, result: RetrievalResult) -> QuestionResult:
    ranked = [h.chunk for h in result.hits]
    found_at: dict[str, int | None] = {}
    for ref in item.refs:
        found_at[ref.label] = next((i for i, c in enumerate(ranked, 1) if ref.matches(c)), None)
    first = next((i for i, c in enumerate(ranked, 1) if any(r.matches(c) for r in item.refs)), None)
    recovered = _recovered_by_expansion(item.refs, ranked[:8], [h.chunk for h in result.expansions])
    return QuestionResult(
        id=item.id,
        category=item.category.value,
        ranked=[c.chunk_id for c in ranked],
        first_relevant_rank=first,
        refs_found_at=found_at,
        expansion_ids=[h.chunk.chunk_id for h in result.expansions],
        expansion_recovered=recovered,
        top_score=result.top_score,
    )


def _recovered_by_expansion(
    refs: list[Ref], top: list[ChunkRecord], expansions: list[ChunkRecord]
) -> list[str]:
    return [
        r.label
        for r in refs
        if not any(r.matches(c) for c in top) and any(r.matches(c) for c in expansions)
    ]


def aggregate(results: Sequence[QuestionResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in KS:
        recalls = [
            sum(1 for r in q.refs_found_at.values() if r is not None and r <= k)
            / len(q.refs_found_at)
            for q in results
        ]
        hits = [q.first_relevant_rank is not None and q.first_relevant_rank <= k for q in results]
        metrics[f"recall@{k}"] = statistics.fmean(recalls)
        metrics[f"hit@{k}"] = statistics.fmean(hits)
    metrics["mrr@20"] = statistics.fmean(
        1 / q.first_relevant_rank if q.first_relevant_rank else 0.0 for q in results
    )
    # Recall@8 counting definition / cross-reference expansions as retrieved.
    metrics["recall@8+exp"] = statistics.fmean(
        (
            sum(1 for r in q.refs_found_at.values() if r is not None and r <= 8)
            + len(q.expansion_recovered)
        )
        / len(q.refs_found_at)
        for q in results
    )
    return metrics


def run(
    retriever: Retriever, items: list[GoldenItem], variants: list[Variant]
) -> dict[str, object]:
    answerable = [i for i in items if not i.should_refuse]
    unanswerable = [i for i in items if i.should_refuse]
    report: dict[str, object] = {}
    for v in variants:
        per_q = []
        for item in answerable:
            res = retriever.retrieve(
                item.question,
                mode=v.mode,
                rerank=v.rerank,
                top_n=DEPTH,
                recital_policy=v.recitals,
            )
            per_q.append(score_question(item, res))
        entry: dict[str, object] = {
            "metrics": aggregate(per_q),
            "questions": [asdict(q) for q in per_q],
        }
        by_cat: dict[str, dict[str, float]] = {}
        for cat in sorted({q.category for q in per_q}):
            subset = [q for q in per_q if q.category == cat]
            agg = aggregate(subset)
            by_cat[cat] = {"n": len(subset), "recall@8": agg["recall@8"], "mrr@20": agg["mrr@20"]}
        entry["by_category"] = by_cat
        if v.rerank:
            neg = [
                retriever.retrieve(
                    i.question, mode=v.mode, rerank=True, recital_policy=v.recitals
                ).top_score
                or 0.0
                for i in unanswerable
            ]
            pos = [q.top_score or 0.0 for q in per_q]
            entry["top_score"] = {
                "answerable": _describe(pos),
                "unanswerable": _describe(neg),
                "unanswerable_by_id": {
                    i.id: round(s, 4) for i, s in zip(unanswerable, neg, strict=True)
                },
            }
        report[v.name] = entry
    return report


def _describe(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    return {
        "min": round(s[0], 4),
        "p10": round(s[max(0, int(len(s) * 0.1) - 1)], 4),
        "median": round(statistics.median(s), 4),
        "max": round(s[-1], 4),
    }


def markdown_table(report: dict[str, object]) -> str:
    cols = ["hit@1", "recall@3", "recall@5", "recall@8", "recall@8+exp", "recall@20", "mrr@20"]
    lines = ["| variant | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, entry in report.items():
        m = entry["metrics"]  # type: ignore[index]
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args(argv)

    settings = Settings()
    retriever = Retriever.from_settings(settings)
    corpus = list(retriever.searcher.corpus)
    items = load_golden(args.golden)
    if bad := unresolved_refs(items, corpus):
        raise SystemExit(f"golden refs not in corpus: {bad}")

    variants = [VARIANTS[v] for v in args.variants.split(",")]
    report = run(retriever, items, variants)

    meta = {
        "run_at": datetime.now(UTC).isoformat(),
        "corpus_hash": corpus_hash(corpus),
        "retriever_config_hash": config_hash(settings.retriever),
        "retriever_config": settings.retriever.model_dump(mode="json"),
        "n_answerable": sum(1 for i in items if not i.should_refuse),
        "n_unanswerable": sum(1 for i in items if i.should_refuse),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "retrieval_latest.json"
    out.write_text(json.dumps({"meta": meta, "variants": report}, indent=2), "utf-8")
    table = markdown_table(report)
    (RESULTS_DIR / "retrieval_latest.md").write_text(table + "\n", "utf-8")

    print(table)
    for name, entry in report.items():
        if "top_score" in entry:  # type: ignore[operator]
            print(f"\n{name} top rerank score: {json.dumps(entry['top_score'])}")  # type: ignore[index]
        print(f"\n{name} by category: {json.dumps(entry['by_category'])}")  # type: ignore[index]
        if args.show_misses:
            for q in entry["questions"]:  # type: ignore[index]
                missing = [r for r, rank in q["refs_found_at"].items() if rank is None or rank > 8]
                if missing:
                    print(f"  MISS {q['id']} {missing} top3={q['ranked'][:3]}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
