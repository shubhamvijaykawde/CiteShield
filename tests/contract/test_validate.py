"""Fault injection: each deliberately broken record must land in quarantine with a reason."""

from __future__ import annotations

import copy
from typing import Any

from citeshield.config import IngestConfig
from citeshield.contracts.chunk import sha256_text
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.quarantine import reason_counts
from citeshield.ingest.validate import validate_document


def _find(records: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    return next(r for r in records if str(r["chunk_id"]).split("@")[0].endswith(suffix))


def _set_text(record: dict[str, Any], text: str) -> None:
    from citeshield.contracts.chunk import estimate_tokens

    record.update(text=text, content_sha256=sha256_text(text), n_tokens=estimate_tokens(text))


def test_clean_fixture_passes(parsed: ParseResult, cfg: IngestConfig) -> None:
    result = validate_document(parsed.records, parsed.toc, cfg)
    assert result.quarantined == []
    assert result.doc_errors == []
    assert len(result.valid) == len(parsed.records)


def test_schema_failure_is_quarantined(parsed: ParseResult, cfg: IngestConfig) -> None:
    records = copy.deepcopy(parsed.records)
    _find(records, "art1")["content_sha256"] = "0" * 64
    result = validate_document(records, parsed.toc, cfg)
    [entry] = result.quarantined
    assert entry.stage == "schema"
    assert "content_sha256" in entry.errors[0]


def test_paragraph_text_number_mismatch(parsed: ParseResult, cfg: IngestConfig) -> None:
    records = copy.deepcopy(parsed.records)
    _set_text(_find(records, "art2:p2"), "3. Paragraph 1 shall not apply to research widgets.")
    result = validate_document(records, parsed.toc, cfg)
    [entry] = result.quarantined
    assert entry.stage == "semantic"
    assert "does not start with paragraph number '2.'" in entry.errors[0]


def test_article_not_in_toc(parsed: ParseResult, cfg: IngestConfig) -> None:
    toc = copy.deepcopy(parsed.toc)
    toc.articles = [2]
    result = validate_document(parsed.records, toc, cfg)
    assert {e.stage for e in result.quarantined} == {"semantic"}
    assert "missing article numbers: [1]" in result.doc_errors


def test_paragraph_gap_quarantines_the_whole_article(
    parsed: ParseResult, cfg: IngestConfig
) -> None:
    toc = copy.deepcopy(parsed.toc)
    toc.paragraphs[2] = [1, 3]  # parser "lost" paragraph 2
    records = [r for r in parsed.records if r["paragraph"] != 2]
    result = validate_document(records, toc, cfg)
    quarantined = {str(e.record["chunk_id"]) for e in result.quarantined}
    assert quarantined and all(":art2:" in cid for cid in quarantined)
    assert all(e.stage == "collection" for e in result.quarantined)


def test_duplicates_are_all_quarantined(parsed: ParseResult, cfg: IngestConfig) -> None:
    dup = copy.deepcopy(_find(parsed.records, "art1"))
    result = validate_document([*parsed.records, dup], parsed.toc, cfg)
    assert len(result.quarantined) == 2
    assert reason_counts(result.quarantined) == {
        "collection: duplicate chunk_id (# records share it)": 2
    }


def test_token_bounds(parsed: ParseResult) -> None:
    tight = IngestConfig(hard_max_chunk_tokens=20)
    result = validate_document(parsed.records, parsed.toc, tight)
    assert result.quarantined
    assert all("outside [" in e.errors[0] for e in result.quarantined)
    assert 0 < result.quarantine_rate < 1
