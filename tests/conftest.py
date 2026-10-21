from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from citeshield.config import IngestConfig
from citeshield.ingest.manifest import SourceSpec
from citeshield.ingest.parse_xhtml import ParseResult, parse_document

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def src() -> SourceSpec:
    return SourceSpec(
        regulation="Test Reg",
        celex="32099R0001",
        title="Test Regulation",
        version="oj-2099-01-01",
        version_date=date(2099, 1, 1),
    )


@pytest.fixture
def mini_xhtml() -> bytes:
    return (FIXTURES / "mini_act.xhtml").read_bytes()


@pytest.fixture
def cfg() -> IngestConfig:
    return IngestConfig()


@pytest.fixture
def parsed(mini_xhtml: bytes, src: SourceSpec) -> ParseResult:
    # A small token budget forces the point/subparagraph splitting paths on the fixture.
    return parse_document(mini_xhtml, src, max_tokens=40)
