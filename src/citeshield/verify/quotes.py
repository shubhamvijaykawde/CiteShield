# ruff: noqa: RUF001  (the fold table lists typographic quote/dash characters on purpose)
"""Deterministic quote matching against canonical text.

Normalisation is deliberately small and fully deterministic: Unicode NFKC, typographic quotes
and dashes folded to ASCII, soft hyphens removed, whitespace collapsed, case folded, and
leading/trailing punctuation trimmed from each quoted segment. Nothing fuzzy is accepted; the
similarity score is reported only to explain a near miss.

A quote may skip text with an ellipsis ("...", "…", "[...]"). Each segment must then appear
in order and be at least ``min_segment_chars`` long, so a quote cannot be stitched together
from tiny fragments ("the" ... "shall").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "″": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "­": None,
    }
)
_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")
_EDGE_PUNCT = " .,;:'\"()"
MAX_SEGMENTS = 4


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).translate(_FOLD)
    return _WS.sub(" ", folded).strip().casefold()


@dataclass(frozen=True)
class QuoteResult:
    matched: bool
    similarity: float  # longest common block / quote length; diagnostic only
    reason: str | None = None


def match_quote(quote: str, source: str, *, min_segment_chars: int = 12) -> QuoteResult:
    src = normalise(source)
    segments = [normalise(s).strip(_EDGE_PUNCT) for s in _ELLIPSIS.split(quote)]
    segments = [s for s in segments if s]
    whole = normalise(quote)
    similarity = _similarity(whole, src)

    if not segments:
        return QuoteResult(False, similarity, "empty quote")
    if len(segments) > MAX_SEGMENTS:
        return QuoteResult(False, similarity, f"quote has more than {MAX_SEGMENTS} segments")
    short = [s for s in segments if len(s) < min_segment_chars]
    if short:
        return QuoteResult(
            False, similarity, f"quoted segment shorter than {min_segment_chars} characters"
        )
    pos = 0
    for seg in segments:
        found = src.find(seg, pos)
        if found < 0:
            return QuoteResult(False, similarity, "quote not found in the canonical text")
        pos = found + len(seg)
    return QuoteResult(True, similarity)


def _similarity(quote: str, source: str) -> float:
    if not quote:
        return 0.0
    block = SequenceMatcher(None, quote, source, autojunk=False).find_longest_match(
        0, len(quote), 0, len(source)
    )
    return round(block.size / len(quote), 3)
