"""Apply the data contract to parsed records.

Three stages, each of which can quarantine a record:

1. ``schema``     - the record must parse as a ``ChunkRecord`` (types, ids, hashes, provenance).
2. ``semantic``   - the record must agree with the document's own table of contents and with
                    itself (paragraph text starts with its number, token bounds, ...).
3. ``collection`` - checks that need every record in the document (duplicate ids, gaps in
                    paragraph numbering).

Document-level problems (e.g. a missing article) cannot be pinned on one record; they are
reported as ``doc_errors`` and fail the run.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from citeshield.config import IngestConfig
from citeshield.contracts.chunk import ChunkKind, ChunkRecord, estimate_tokens
from citeshield.ingest.parse_xhtml import DocumentToc, RawRecord

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


@dataclass(frozen=True)
class QuarantineEntry:
    stage: str
    errors: list[str]
    record: RawRecord


@dataclass
class ValidationResult:
    valid: list[ChunkRecord] = field(default_factory=list)
    quarantined: list[QuarantineEntry] = field(default_factory=list)
    doc_errors: list[str] = field(default_factory=list)

    @property
    def quarantine_rate(self) -> float:
        total = len(self.valid) + len(self.quarantined)
        return len(self.quarantined) / total if total else 0.0


def validate_document(
    records: list[RawRecord], toc: DocumentToc, cfg: IngestConfig
) -> ValidationResult:
    result = ValidationResult(doc_errors=toc_errors(toc))

    typed: list[tuple[ChunkRecord, RawRecord]] = []
    for raw in records:
        try:
            rec = ChunkRecord.model_validate(raw)
        except ValidationError as e:
            errors = [
                f"{'.'.join(map(str, err['loc'])) or '<record>'}: {err['msg']}"
                for err in e.errors()
            ]
            result.quarantined.append(QuarantineEntry("schema", errors, raw))
            continue
        if errors := semantic_errors(rec, toc, cfg):
            result.quarantined.append(QuarantineEntry("semantic", errors, raw))
            continue
        typed.append((rec, raw))

    id_counts = Counter(rec.chunk_id for rec, _ in typed)
    gappy = {
        art: paras
        for art, paras in toc.paragraphs.items()
        if paras and paras != list(range(1, len(paras) + 1))
    }
    for rec, raw in typed:
        errors = []
        if id_counts[rec.chunk_id] > 1:
            errors.append(f"duplicate chunk_id ({id_counts[rec.chunk_id]} records share it)")
        if rec.kind is ChunkKind.ARTICLE and rec.article in gappy:
            errors.append(
                f"article {rec.article} paragraph numbering is not contiguous: {gappy[rec.article]}"
            )
        if errors:
            result.quarantined.append(QuarantineEntry("collection", errors, raw))
        else:
            result.valid.append(rec)
    return result


def semantic_errors(rec: ChunkRecord, toc: DocumentToc, cfg: IngestConfig) -> list[str]:
    errors = []
    if not cfg.min_chunk_tokens <= rec.n_tokens <= cfg.hard_max_chunk_tokens:
        errors.append(
            f"n_tokens {rec.n_tokens} outside [{cfg.min_chunk_tokens}, {cfg.hard_max_chunk_tokens}]"
        )
    if rec.n_tokens != estimate_tokens(rec.text):
        errors.append("n_tokens does not match text")

    if rec.kind is ChunkKind.ARTICLE:
        assert rec.article is not None
        if rec.article not in toc.articles:
            errors.append(f"article {rec.article} not in document")
        if rec.paragraph is not None:
            if rec.paragraph not in toc.paragraphs.get(rec.article, []):
                errors.append(f"paragraph {rec.article}({rec.paragraph}) not in document")
            # The OJ prints the paragraph number as the first token; a mismatch means the
            # parser attached text to the wrong paragraph.
            if not re.match(rf"^{rec.paragraph}\.\s", rec.text):
                errors.append(f"text does not start with paragraph number '{rec.paragraph}.'")
    elif rec.kind is ChunkKind.RECITAL:
        if rec.recital not in toc.recitals:
            errors.append(f"recital {rec.recital} not in document")
    elif rec.annex not in toc.annexes:
        errors.append(f"annex {rec.annex} not in document")
    return errors


def toc_errors(toc: DocumentToc) -> list[str]:
    errors = []
    if not toc.articles:
        errors.append("document has no articles")
    errors += _sequence_errors("article", toc.articles)
    errors += _sequence_errors("recital", toc.recitals)
    errors += _sequence_errors("annex", [_roman_to_int(a) for a in toc.annexes])
    return errors


def _sequence_errors(name: str, numbers: list[int]) -> list[str]:
    if not numbers:
        return []
    errors = []
    dupes = sorted(n for n, c in Counter(numbers).items() if c > 1)
    if dupes:
        errors.append(f"duplicate {name} numbers: {dupes}")
    missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
    if missing:
        errors.append(f"missing {name} numbers: {missing}")
    return errors


def _roman_to_int(roman: str) -> int:
    total = 0
    for ch, nxt in zip(roman, [*roman[1:], ""], strict=True):
        value = _ROMAN[ch]
        total += -value if nxt and _ROMAN[nxt] > value else value
    return total
