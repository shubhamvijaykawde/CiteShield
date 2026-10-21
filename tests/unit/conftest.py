from __future__ import annotations

import pytest

from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.retrieval.corpus import Corpus
from vfx import verification_corpus


@pytest.fixture
def corpus(parsed: ParseResult) -> Corpus:
    return verification_corpus(parsed)
