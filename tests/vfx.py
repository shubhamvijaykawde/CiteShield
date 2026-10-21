"""Shared builders for the verification tests (fixture corpus, evidence, drafted answers)."""

from __future__ import annotations

from citeshield.agent.tools import EvidenceItem
from citeshield.config import IngestConfig
from citeshield.contracts.answer import AgentAnswer
from citeshield.contracts.chunk import (
    ChunkKind,
    ChunkRecord,
    TrustTier,
    estimate_tokens,
    make_chunk_id,
    sha256_text,
)
from citeshield.ingest.parse_xhtml import ParseResult
from citeshield.ingest.validate import validate_document
from citeshield.retrieval.corpus import Corpus

V = "@oj-2099-01-01"
# P2:  "2. Paragraph 1 shall not apply to widgets used solely for research."
# PTA: "1. Providers shall: (a) keep records of every widget test for ... ten years ..."
P2 = f"32099R0001:art2:p2{V}"
PTA = f"32099R0001:art2:p1:pta{V}"
ART1 = f"32099R0001:art1{V}"
SUPPORTED = (0.95, 0.04, 0.01)
NEUTRAL = (0.05, 0.9, 0.05)
CONTRA = (0.02, 0.03, 0.95)


def untrusted_chunk() -> ChunkRecord:
    text = "Research widgets are exempt from every obligation in this Regulation."
    return ChunkRecord(
        chunk_id=make_chunk_id(
            celex="32099R0001", kind=ChunkKind.ARTICLE, version="blog-1", article=99
        ),
        regulation="Test Reg", celex="32099R0001", kind=ChunkKind.ARTICLE, article=99,
        heading_path=["Test Reg", "Commentary"], text=text, source_url="https://blog.example.com/x",
        version="blog-1", version_date="2099-01-01", content_sha256=sha256_text(text),
        n_tokens=estimate_tokens(text), trust_tier=TrustTier.UNTRUSTED,
    )  # fmt: skip


def verification_corpus(parsed: ParseResult) -> Corpus:
    """The fixture act plus one untrusted (non-citable) commentary chunk."""
    valid = validate_document(parsed.records, parsed.toc, IngestConfig()).valid
    return Corpus([*valid, untrusted_chunk()])


def evidence(corpus: Corpus, *ids: str, score: float | None = 0.9) -> dict[str, EvidenceItem]:
    return {i: EvidenceItem.from_chunk(corpus[i], "retrieve:search", score) for i in ids}


def answer(
    *claims: tuple[str, list[tuple[str, str]]],
    summary: str = "",
    caveats: list[str] | None = None,
    status: str = "answered",
) -> AgentAnswer:
    return AgentAnswer.model_validate(
        {
            "status": status,
            "summary": summary,
            "claims": [
                {"text": t, "citations": [{"chunk_id": c, "quote": q} for c, q in cites]}
                for t, cites in claims
            ],
            "caveats": caveats or [],
        }
    )


EXEMPT = "Paragraph 1 does not apply to widgets used solely for research."
EXEMPT_QUOTE = "shall not apply to widgets used solely for research"
