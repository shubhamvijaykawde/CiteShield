"""Provision-reference checker: resolution against the real corpus, and its uncertainty."""

from __future__ import annotations

from typing import Any

import pytest

from citeshield.config import PROJECT_ROOT
from citeshield.contracts.answer import AgentAnswer
from citeshield.evaluation.provisions import (
    CHECKER_VERSION,
    ProvisionChecker,
    RefStatus,
    confirmed_nonexistent,
)
from citeshield.evaluation.records import AnswerRecord
from citeshield.retrieval.corpus import Corpus
from helpers import make_chunk

CORPUS = Corpus.load(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
PC = ProvisionChecker(CORPUS)
S = RefStatus


def one(sentence: str, answer_regs: set[str] | None = None) -> tuple[str, RefStatus, str | None]:
    (f,) = PC.scan_sentence(sentence, answer_regs or set())
    return f.text, f.status, f.regulation


@pytest.mark.parametrize(
    "sentence",
    [
        "Article 16 of the AI Act lists the obligations.",
        "AI Act Article 16 lists the obligations.",
        "Under Article 16(a) of the AI Act providers ensure compliance.",  # point in the text
        "GDPR Article 6(1)(f) covers legitimate interests.",
        "Art. 4(11) GDPR defines consent.",  # numbered definition point
        "AI Act Art. 5(1)(h) covers remote biometric identification.",  # split point
        "Article 6 of Regulation (EU) 2016/679 sets the lawful bases.",  # CELEX-derived alias
        "Article 23(4) of the NIS 2 Directive sets the deadlines.",
        "Article 33 paragraph 1 GDPR sets the 72-hour deadline.",
    ],
)
def test_existing_provisions(sentence: str) -> None:
    assert one(sentence)[1] is S.EXISTS


@pytest.mark.parametrize(
    ("sentence", "regs", "reg"),
    [
        ("AI Act Article 16(2) requires X.", None, "AI Act"),
        # The a5 answer: no regulation in the sentence, the answer covers only the AI Act.
        ("These duties are detailed in Article 16(2).", {"AI Act"}, "AI Act"),
        ("Article 5(1)(z) of the AI Act.", None, "AI Act"),  # Art. 5(1) splits its points
        ("Art. 4(40) GDPR.", None, "GDPR"),  # Art. 4 numbers its points 1-26
        ("GDPR Article 33 paragraph 6 applies.", None, "GDPR"),
        ("Article 120 GDPR.", None, "GDPR"),
    ],
)
def test_confirmed_nonexistent(sentence: str, regs: set[str] | None, reg: str) -> None:
    _, status, regulation = one(sentence, regs)
    assert status is S.NONEXISTENT and regulation == reg


@pytest.mark.parametrize(
    ("sentence", "status"),
    [
        ("Article 16.2 of the AI Act.", S.UNPARSEABLE),
        ("Article 16-2 of the AI Act.", S.UNPARSEABLE),
        ("Article 16(zz9) of the AI Act.", S.UNPARSEABLE),
        ("Article 5 of Directive 2016/2102 applies.", S.OUT_OF_CORPUS),
        # Art. 16 does not split out its points and has no "(z)" marker: undecidable.
        ("AI Act Article 16(z) applies.", S.AMBIGUOUS),
    ],
)
def test_fake_or_undecidable_formats_are_never_confirmed(sentence: str, status: RefStatus) -> None:
    assert one(sentence)[1] is status


def test_non_article_formats_are_not_references() -> None:
    assert PC.scan_sentence("See § 16(2) AI Act and part 16(2) of the guide.") == []


@pytest.mark.parametrize(
    ("sentence", "regs", "why"),
    [
        ("Article 99(2) applies.", set(), "no regulation named"),
        ("Article 33(1) applies.", {"GDPR", "NIS2"}, "several regulations in the answer"),
        (
            "Under the GDPR and NIS2, Article 33(1) applies.",
            set(),
            "several regulations in the sentence",
        ),
    ],
)
def test_unknown_regulation_is_ambiguous(sentence: str, regs: set[str], why: str) -> None:
    (f,) = PC.scan_sentence(sentence, regs)
    assert f.status is S.AMBIGUOUS and f.reason == why


def test_discussing_an_invalid_reference_is_not_asserting_it() -> None:
    findings = PC.scan_sentence(
        "The source mentions Article 16(2), but there is no Article 16(2) in the AI Act."
    )
    assert {f.status for f in findings} == {S.DISPUTED}
    assert confirmed_nonexistent(findings) == []


def test_unsplit_article_with_a_marker_for_the_number_is_ambiguous() -> None:
    # An article stored as one chunk whose text shows a "(2)" sub-unit: cannot rule it out.
    chunk = make_chunk("GDPR", 7, "Text of the article:\n(1) first item;\n(2) second item.")
    pc = ProvisionChecker(Corpus([chunk]))
    assert pc.resolve("GDPR", 7, ["2"])[0] is S.AMBIGUOUS
    assert pc.resolve("GDPR", 7, ["3"])[0] is S.NONEXISTENT


def _record(answer: AgentAnswer | None, error: str | None = None) -> AnswerRecord:
    return AnswerRecord.model_validate(
        {
            "id": "q087",
            "category": "checklist",
            "question": "Q?",
            "should_refuse": False,
            "expected": ["AI Act 16"],
            "reference_answer": "Ref.",
            "answer": answer,
            "error": error,
            "schema_valid": answer is not None,
        }
    )


def _answer(**kw: Any) -> AgentAnswer:
    cite = [
        {
            "chunk_id": "32024R1689:art16@oj-2024-07-12",
            "quote": kw.pop("quote", "Providers of high-risk AI systems shall"),
        }
    ]
    return AgentAnswer.model_validate(
        {
            "status": "answered",
            "summary": kw.pop("summary", "Providers have obligations."),
            "claims": [{"text": kw.pop("claim", "Providers must comply."), "citations": cite}],
            "checklist": kw.pop("checklist", None),
            "caveats": kw.pop("caveats", []),
        }
    )


def test_scan_covers_summary_claims_checklist_and_caveats_but_not_quotes() -> None:
    answer = _answer(
        summary="Under the AI Act, providers must comply. These duties are in Article 16(2).",
        claim="See Art. 16(3).",
        checklist=[
            {
                "obligation": "Follow Article 16(4).",
                "applies_to": "providers",
                "citations": [
                    {
                        "chunk_id": "32024R1689:art16@oj-2024-07-12",
                        "quote": "Providers of high-risk AI systems shall",
                    }
                ],
            }
        ],
        caveats=["Article 16(5) may apply."],
        quote="Article 16(9) is quoted here, verbatim",
    )
    found = confirmed_nonexistent(PC.scan(_record(answer)))
    assert found == ["Article 16(2)", "Article 16(5)", "Art. 16(3)", "Article 16(4)"]


def test_cited_regulation_resolves_unnamed_references() -> None:
    # No regulation in the text at all; the only cited chunk is AI Act Art. 16.
    rec = _record(_answer(summary="The obligations are in Article 16(2)."))
    (f,) = PC.scan(rec)
    assert f.status is S.NONEXISTENT and f.regulation == "AI Act" and f.basis == "answer"


def test_malformed_or_missing_answer_has_no_findings() -> None:
    assert PC.scan(_record(None, error="ReadTimeout: timed out")) == []


def test_checker_identity_is_stable_and_versioned() -> None:
    meta = PC.meta()
    assert meta["version"] == CHECKER_VERSION
    assert meta == ProvisionChecker(CORPUS).meta()
    assert meta["corpus"] == "b5afd60b188d1cfa"
    other = ProvisionChecker(Corpus([make_chunk("GDPR", 7, "Some article text here.")]))
    assert other.meta()["config_sha"] != meta["config_sha"]  # different alias table
