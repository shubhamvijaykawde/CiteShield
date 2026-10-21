"""The ChunkRecord contract rejects every malformed shape we know how to produce."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from citeshield.contracts.chunk import ChunkRecord, sha256_text
from citeshield.ingest.parse_xhtml import ParseResult


@pytest.fixture
def good(parsed: ParseResult) -> dict[str, Any]:
    return dict(next(r for r in parsed.records if r["paragraph"] == 2))


def _with(record: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return {**record, **changes}


def test_good_record_round_trips(good: dict[str, Any]) -> None:
    rec = ChunkRecord.model_validate(good)
    assert ChunkRecord.model_validate_json(rec.model_dump_json()) == rec
    assert rec.citation_label == "Test Reg Art. 2(2)"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unexpected": 1}, "Extra inputs are not permitted"),
        ({"chunk_id": "32099R0001:art2:p3@oj-2099-01-01"}, "chunk_id"),
        ({"text": "Tampered text that no longer matches its hash."}, "content_sha256"),
        ({"source_url": "https://example.com/gdpr#art_2"}, "non-official host"),
        ({"article": None}, "missing its article"),
        ({"recital": 4}, "exactly one of"),
        ({"point": "(a)"}, "point"),
        ({"celex": "2016/679"}, "celex"),
        ({"text": "Too short", "content_sha256": sha256_text("Too short")}, "text"),
        ({"language": "de"}, "language"),
    ],
)
def test_malformed_records_are_rejected(
    good: dict[str, Any], changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ChunkRecord.model_validate(_with(good, **changes))


def test_untrusted_tier_may_come_from_any_host(good: dict[str, Any]) -> None:
    rec = ChunkRecord.model_validate(
        _with(good, source_url="https://blog.example.com/x", trust_tier="untrusted")
    )
    assert rec.trust_tier.value == "untrusted"


def test_recital_cannot_carry_a_paragraph(parsed: ParseResult) -> None:
    recital = dict(next(r for r in parsed.records if r["kind"] == "recital"))
    with pytest.raises(ValidationError, match="only valid on article chunks"):
        ChunkRecord.model_validate(_with(recital, paragraph=1))


@pytest.mark.parametrize(
    ("loc", "label"),
    [
        ({"paragraph": 1, "point": "a"}, "Test Reg Art. 2(1)(a)"),
        (
            {"paragraph": 1, "subparagraph": 2, "point": "a"},
            "Test Reg Art. 2(1), subpara. 2, point (a)",
        ),
    ],
)
def test_citation_labels(parsed: ParseResult, loc: dict[str, Any], label: str) -> None:
    raw = next(
        r
        for r in parsed.records
        if r["article"] == 2
        and all(r[k] == v for k, v in loc.items())
        and (("subparagraph" in loc) or r["subparagraph"] is None)
    )
    assert ChunkRecord.model_validate(raw).citation_label == label
