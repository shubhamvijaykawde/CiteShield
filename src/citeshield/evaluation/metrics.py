"""Deterministic answer metrics over a list of ``AnswerRecord``.

Paragraph-level matching uses the golden refs (``golden.Ref.matches``), so a citation of
GDPR 6(1)(a) counts for an expected "GDPR 6(1)".

Crashes (no answer at all: context overflow, server error, ...) are infrastructure failures,
not agent behaviour. They are counted in ``n_crashed`` / ``error_rate`` and in schema
validity, and are excluded from every quality metric below, so a crash can neither pass as
a correct refusal nor drag down citation scores. Quality metrics are over completed items.

* citation precision: share of an answer's distinct cited chunks that match an expected
  ref; averaged over answerable items that cite something.
* citation recall: share of an item's expected refs matched by some cited chunk; averaged
  over completed answerable items (a refusal scores 0).
* refusal precision / recall / F1: a deliberate refusal is the positive class.
* schema validity: share of all items whose final answer round-trips ``AgentAnswer``
  (crashes count as invalid). first-draft validity: share of completed items whose
  first draft needed no repair.
* faithfulness: mean verifier support ratio of the draft behind each answer.
* context precision: RAGAS-style rank-weighted precision of the retrieved chunks (mean of
  precision@k at the ranks of relevant chunks); context recall: share of expected refs
  that some retrieved chunk matches. Retrieval order is the order tools returned chunks.
* latency p50/p95: nearest-rank quantiles over completed items. cost: mean per item.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

from citeshield.contracts.chunk import ChunkRecord
from citeshield.evaluation.golden import Ref, parse_ref
from citeshield.evaluation.records import AnswerRecord

Lookup = Callable[[str], ChunkRecord | None]
Metrics = dict[str, float | None]

# Metric names in report order. The gate and the baseline files use these keys.
METRIC_KEYS = (
    "n",
    "n_completed",
    "n_crashed",
    "citation_precision",
    "citation_recall",
    "refusal_precision",
    "refusal_recall",
    "refusal_f1",
    "schema_validity",
    "first_draft_validity",
    "faithfulness",
    "context_precision",
    "context_recall",
    "error_rate",
    "latency_p50_s",
    "latency_p95_s",
    "cost_per_question_usd",
    "judge_n",
    "judge_correctness",
    "judge_faithful",
    "judge_citation_adequate",
)


def _matches(chunk_id: str, refs: Sequence[Ref], lookup: Lookup) -> bool:
    chunk = lookup(chunk_id)
    return chunk is not None and any(r.matches(chunk) for r in refs)


def _covered(refs: Sequence[Ref], ids: Sequence[str], lookup: Lookup) -> float:
    chunks = [c for c in map(lookup, ids) if c is not None]
    return sum(1 for r in refs if any(r.matches(c) for c in chunks)) / len(refs)


def citation_precision(rec: AnswerRecord, lookup: Lookup) -> float | None:
    if rec.should_refuse or not rec.cited_ids:
        return None
    refs = [parse_ref(r) for r in rec.expected]
    return sum(_matches(i, refs, lookup) for i in rec.cited_ids) / len(rec.cited_ids)


def citation_recall(rec: AnswerRecord, lookup: Lookup) -> float | None:
    if rec.should_refuse:
        return None
    return _covered([parse_ref(r) for r in rec.expected], rec.cited_ids, lookup)


def context_precision(rec: AnswerRecord, lookup: Lookup) -> float | None:
    if rec.should_refuse or not rec.retrieved:
        return None
    refs = [parse_ref(r) for r in rec.expected]
    hits, precisions = 0, []
    for k, cid in enumerate(rec.retrieved, 1):
        if _matches(cid, refs, lookup):
            hits += 1
            precisions.append(hits / k)
    return statistics.fmean(precisions) if precisions else 0.0


def context_recall(rec: AnswerRecord, lookup: Lookup) -> float | None:
    if rec.should_refuse:
        return None
    return _covered([parse_ref(r) for r in rec.expected], rec.retrieved, lookup)


def nearest_rank(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else None


def _share(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def refusal_scores(records: Sequence[AnswerRecord]) -> tuple[float, float, float]:
    tp = sum(1 for r in records if r.refused and r.should_refuse)
    fp = sum(1 for r in records if r.refused and not r.should_refuse)
    fn = sum(1 for r in records if not r.refused and r.should_refuse)
    # No refusals predicted or expected: nothing went wrong, so precision/recall are 1.
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def compute(
    records: Sequence[AnswerRecord],
    lookup: Lookup,
    judged: dict[str, Any] | None = None,
) -> Metrics:
    """Aggregate metrics. ``judged`` maps record id to a ``JudgeVerdict`` (or None)."""
    ok = [r for r in records if not r.crashed]
    p, rc, f1 = refusal_scores(ok)
    latencies = [r.latency_ms / 1000 for r in ok]
    verdicts = [v for v in (judged or {}).values() if v is not None]
    out: Metrics = {
        "n": float(len(records)),
        "n_completed": float(len(ok)),
        "n_crashed": float(len(records) - len(ok)),
        "citation_precision": _mean([citation_precision(r, lookup) for r in ok]),
        "citation_recall": _mean([citation_recall(r, lookup) for r in ok]),
        "refusal_precision": p,
        "refusal_recall": rc,
        "refusal_f1": f1,
        "schema_validity": _share([r.schema_valid for r in records]),
        "first_draft_validity": _share([r.first_draft_valid for r in ok]),
        "faithfulness": _mean([r.support_ratio for r in ok]),
        "context_precision": _mean([context_precision(r, lookup) for r in ok]),
        "context_recall": _mean([context_recall(r, lookup) for r in ok]),
        "error_rate": _share([r.crashed for r in records]),
        "latency_p50_s": nearest_rank(latencies, 0.50),
        "latency_p95_s": nearest_rank(latencies, 0.95),
        "cost_per_question_usd": _mean([r.cost_usd for r in records]),
        "judge_n": float(len(verdicts)),
        "judge_correctness": _mean([float(v.correctness) for v in verdicts]),
        "judge_faithful": _share([v.faithful for v in verdicts]),
        "judge_citation_adequate": _share([v.citation_adequate for v in verdicts]),
    }
    return out


def by_category(records: Sequence[AnswerRecord], lookup: Lookup) -> dict[str, Metrics]:
    groups: dict[str, list[AnswerRecord]] = {}
    for r in records:
        groups.setdefault(r.category.value, []).append(r)
    return {
        cat: {
            "n": float(len(rs)),
            "crashed": float(sum(r.crashed for r in rs)),
            # Shares below are over the category's completed items.
            "refused": _share([r.refused for r in ok]),
            "citation_precision": _mean([citation_precision(r, lookup) for r in ok]),
            "citation_recall": _mean([citation_recall(r, lookup) for r in ok]),
            "faithfulness": _mean([r.support_ratio for r in ok]),
        }
        for cat, rs in sorted(groups.items())
        for ok in [[r for r in rs if not r.crashed]]
    }
