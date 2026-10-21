from __future__ import annotations

from datetime import date

import pytest

from citeshield.config import PROJECT_ROOT
from citeshield.contracts.chunk import (
    ChunkKind,
    ChunkRecord,
    TrustTier,
    estimate_tokens,
    make_chunk_id,
    sha256_text,
)
from citeshield.evaluation.golden import load_golden, parse_ref
from citeshield.evaluation.retrieval import QuestionResult, aggregate


def chunk(**loc: object) -> ChunkRecord:
    text = str(loc.pop("text", "1. Some sufficiently long legal text for the test."))
    kind = ChunkKind(str(loc.pop("kind", "article")))
    return ChunkRecord.model_validate(
        {
            "chunk_id": make_chunk_id(celex="32016R0679", kind=kind, version="v", **loc),  # type: ignore[arg-type]
            "regulation": "GDPR",
            "celex": "32016R0679",
            "kind": kind,
            **loc,
            "heading_path": ["GDPR", "x"],
            "text": text,
            "source_url": "https://eur-lex.europa.eu/x",
            "version": "v",
            "version_date": date(2016, 5, 4),
            "content_sha256": sha256_text(text),
            "n_tokens": estimate_tokens(text),
            "trust_tier": TrustTier.OFFICIAL,
        }
    )


@pytest.mark.parametrize(
    ("ref", "loc", "expected"),
    [
        ("GDPR 22(1)", {"article": 22, "paragraph": 1}, True),
        ("GDPR 22(1)", {"article": 22, "paragraph": 1, "point": "a"}, True),
        ("GDPR 22(1)", {"article": 22, "paragraph": 2}, False),
        ("GDPR 22", {"article": 22, "paragraph": 3}, True),
        ("GDPR 5(1)(a)", {"article": 5, "paragraph": 1, "point": "b"}, False),
        ("GDPR 4(5)", {"article": 4, "point": "5"}, True),  # definitions are numbered points
        ("AI Act 6(1)", {"article": 6, "paragraph": 1}, False),  # wrong regulation
        (
            "GDPR Recital 71",
            {"kind": "recital", "recital": 71, "text": "Recital text, long enough."},
            True,
        ),
    ],
)
def test_ref_matching(ref: str, loc: dict[str, object], expected: bool) -> None:
    assert parse_ref(ref).matches(chunk(**loc)) is expected


def test_annex_item_matching() -> None:
    ref = parse_ref("GDPR Annex II.4")
    assert ref.matches(chunk(kind="annex", annex="II", part=3, text="4. Employment, workers: ..."))
    assert not ref.matches(
        chunk(kind="annex", annex="II", part=4, text="5. Access to services ...")
    )


def test_unparseable_ref() -> None:
    with pytest.raises(ValueError, match="cannot parse"):
        parse_ref("Article 22 of something")


def test_golden_file_is_well_formed() -> None:
    items = load_golden(PROJECT_ROOT / "eval" / "golden" / "questions.jsonl")
    assert len(items) >= 50
    assert any(i.should_refuse for i in items)


def test_aggregate_metrics() -> None:
    qs = [
        QuestionResult("q1", "lookup", [], 1, {"A": 1, "B": 10}),
        QuestionResult("q2", "lookup", [], None, {"C": None}, expansion_recovered=["C"]),
    ]
    m = aggregate(qs)
    assert m["hit@1"] == 0.5
    assert m["recall@8"] == pytest.approx(0.25)  # (1/2 + 0) / 2
    assert m["recall@20"] == pytest.approx(0.5)  # B (rank 10) now counts: (2/2 + 0) / 2
    assert m["recall@8+exp"] == pytest.approx(0.75)
    assert m["mrr@20"] == 0.5
