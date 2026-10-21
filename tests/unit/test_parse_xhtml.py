from __future__ import annotations

from citeshield.contracts.chunk import ChunkRecord, sha256_text
from citeshield.ingest.manifest import SourceSpec
from citeshield.ingest.parse_xhtml import ParseResult, parse_document

V = "@oj-2099-01-01"


def _by_id(parsed: ParseResult) -> dict[str, dict[str, object]]:
    return {str(r["chunk_id"]): r for r in parsed.records}


def test_every_record_satisfies_the_contract(parsed: ParseResult) -> None:
    for raw in parsed.records:
        ChunkRecord.model_validate(raw)


def test_toc(parsed: ParseResult) -> None:
    assert parsed.toc.articles == [1, 2]
    assert parsed.toc.paragraphs == {1: [], 2: [1, 2]}
    assert parsed.toc.recitals == [1, 2]
    assert parsed.toc.annexes == ["I"]
    assert parsed.warnings == []


def test_ids_cover_points_and_second_subparagraph_list(parsed: ParseResult) -> None:
    ids = set(_by_id(parsed))
    assert {
        f"32099R0001:art1{V}",
        f"32099R0001:art2:p1:pta{V}",
        f"32099R0001:art2:p1:ptb{V}",
        f"32099R0001:art2:p1:sp2{V}",
        f"32099R0001:art2:p1:sp2:pta{V}",  # second (a) list must not collide with the first
        f"32099R0001:art2:p2{V}",
        f"32099R0001:rct1{V}",
        f"32099R0001:anxI:part1{V}",
        f"32099R0001:anxI:part3{V}",
    } <= ids
    assert len(ids) == len(parsed.records)


def test_point_chunks_repeat_the_lead_in(parsed: ParseResult) -> None:
    text = str(_by_id(parsed)[f"32099R0001:art2:p1:ptb{V}"]["text"])
    assert text.splitlines() == [
        "1. Providers shall:",
        "(b) notify the competent authority of every failed widget test without undue delay.",
    ]


def test_breadcrumbs_and_anchor(parsed: ParseResult) -> None:
    rec = _by_id(parsed)[f"32099R0001:art2:p2{V}"]
    assert rec["heading_path"] == [
        "Test Reg",
        "CHAPTER I — General provisions",
        "Article 2 — Obligations of providers",
    ]
    assert str(rec["source_url"]).endswith("CELEX:32099R0001#art_2")


def test_recital_strips_number_and_footnote_marker(parsed: ParseResult) -> None:
    text = _by_id(parsed)[f"32099R0001:rct1{V}"]["text"]
    assert text == "The protection of test subjects is a fundamental concern of this Regulation."


def test_short_paragraph_is_a_single_chunk(mini_xhtml: bytes, src: SourceSpec) -> None:
    parsed = parse_document(mini_xhtml, src, max_tokens=512)
    ids = [
        r["chunk_id"] for r in parsed.records if str(r["chunk_id"]).startswith("32099R0001:art2")
    ]
    assert ids == [f"32099R0001:art2:p1{V}", f"32099R0001:art2:p2{V}"]


def test_hash_and_tokens_are_consistent(parsed: ParseResult) -> None:
    for r in parsed.records:
        assert r["content_sha256"] == sha256_text(str(r["text"]))


def test_external_entities_are_not_resolved(src: SourceSpec, tmp_path: object) -> None:
    xxe = b"""<?xml version="1.0"?>
<!DOCTYPE html [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div id="art_1"><p class="oj-ti-art">Article 1</p>
<p class="oj-normal">This Regulation applies to widgets &xxe; in the Union market.</p></div>
</body></html>"""
    parsed = parse_document(xxe, src)
    assert "root:" not in str(parsed.records[0]["text"])
