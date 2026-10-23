"""Coordinated claims: every part must be entailed, not only the sentence as a whole.

Regression for the Milestone 7 audit: "Providers must send timely confirmation of receipt
and notify decisions with redress options" was accepted on DSA Art. 16(5), which covers
only the second part (receipt confirmation is Art. 16(4)). The NLI double below reproduces
that: it entails the whole sentence but not the unsupported part.
"""

from __future__ import annotations

import pytest

from citeshield.agent.policy import finalize
from citeshield.contracts.answer import AgentAnswer, AnswerStatus
from citeshield.contracts.verification import ClaimVerdict, VerificationReport
from citeshield.retrieval.corpus import Corpus
from citeshield.verify.config import VerifierConfig
from citeshield.verify.verifier import Verifier, split_conjuncts
from helpers import RuleNLI
from vfx import EXEMPT_QUOTE, NEUTRAL, P2, PTA, SUPPORTED, answer, evidence

EXEMPT_PART = "Paragraph 1 does not apply to widgets used solely for research"
RECORDS_PART = "providers must keep records of every widget test for ten years"
COMPOUND = f"{EXEMPT_PART} and {RECORDS_PART}."
RECORDS_QUOTE = "keep records of every widget test for a period of ten years"


@pytest.mark.parametrize(
    ("text", "parts"),
    [
        (
            "Providers must send timely confirmation of receipt and notify decisions with "
            "redress options.",
            [
                "Providers must send timely confirmation of receipt",
                "Providers must notify decisions with redress options.",
            ],
        ),
        (
            "Data subjects can withdraw consent at any time and withdrawal must be as easy as "
            "giving consent.",
            [
                "Data subjects can withdraw consent at any time",
                "withdrawal must be as easy as giving consent.",
            ],
        ),
        (
            "The controller shall notify the authority; the processor shall inform the controller.",
            [
                "The controller shall notify the authority",
                "the processor shall inform the controller.",
            ],
        ),
        (COMPOUND, [EXEMPT_PART, f"{RECORDS_PART}."]),
    ],
)
def test_split_conjuncts(text: str, parts: list[str]) -> None:
    assert split_conjuncts(text) == parts


@pytest.mark.parametrize(
    "text",
    [
        "Controllers and processors must keep records of processing activities.",
        "Controllers of large online platforms and processors must keep records.",
        "Consent must be freely given, specific, informed and unambiguous.",
        "Processing must be lawful, fair and transparent.",
        "NIS2 requires an early warning in 24 hours and an incident notification in 72 hours.",
        "GDPR Article 6(1)(a) and (b) set out consent and contract.",
    ],
)
def test_noun_coordination_is_not_split(text: str) -> None:
    assert split_conjuncts(text) == []


# The double entails the whole compound and the exemption part, not the records part.
COMPOUND_NLI = {COMPOUND: SUPPORTED, EXEMPT_PART: SUPPORTED, RECORDS_PART: NEUTRAL}


def _verify(
    corpus: Corpus, *cites: tuple[str, str], check: bool = True
) -> tuple[VerificationReport, AgentAnswer]:
    nli = RuleNLI(dict(COMPOUND_NLI))
    verifier = Verifier(corpus, nli, VerifierConfig(check_conjuncts=check))
    draft = answer((COMPOUND, list(cites)))
    return verifier.verify(draft, evidence(corpus, P2, PTA)), draft


def test_whole_sentence_entailment_alone_used_to_pass(corpus: Corpus) -> None:
    # Reproduces the defect: without the part check the compound claim is "supported".
    report, _ = _verify(corpus, (P2, EXEMPT_QUOTE), check=False)
    assert report.claims[0].verdict is ClaimVerdict.SUPPORTED


def test_unsupported_part_fails_the_claim(corpus: Corpus) -> None:
    report, draft = _verify(corpus, (P2, EXEMPT_QUOTE))
    claim = report.claims[0]
    assert claim.verdict is ClaimVerdict.UNSUPPORTED
    assert any("part not entailed" in r and "keep records" in r for r in claim.reasons)
    final = finalize(draft, report, 0.30)
    assert final.status is AnswerStatus.REFUSED  # the only claim is removed, not kept


def test_each_part_supported_by_its_own_citation_passes(corpus: Corpus) -> None:
    rules = {COMPOUND: SUPPORTED, EXEMPT_PART: SUPPORTED, RECORDS_PART: SUPPORTED}
    verifier = Verifier(corpus, RuleNLI(rules))
    draft = answer((COMPOUND, [(P2, EXEMPT_QUOTE), (PTA, RECORDS_QUOTE)]))
    report = verifier.verify(draft, evidence(corpus, P2, PTA))
    assert report.claims[0].verdict is ClaimVerdict.SUPPORTED


def test_parts_are_only_checked_against_citations_that_passed(corpus: Corpus) -> None:
    # PTA's quote fails, so PTA's text is never used as a premise, for the parts either.
    rules = {COMPOUND: SUPPORTED, EXEMPT_PART: SUPPORTED, RECORDS_PART: SUPPORTED}
    nli = RuleNLI(rules)
    draft = answer((COMPOUND, [(P2, EXEMPT_QUOTE), (PTA, "a quote that is not in the text")]))
    Verifier(corpus, nli).verify(draft, evidence(corpus, P2, PTA))
    pta_text = corpus[PTA].text
    assert nli.pairs and all(pta_text not in premise for premise, _ in nli.pairs)


def test_single_clause_claims_are_unchanged(corpus: Corpus) -> None:
    nli = RuleNLI({EXEMPT_PART: SUPPORTED})
    report = Verifier(corpus, nli).verify(
        answer((EXEMPT_PART + ".", [(P2, EXEMPT_QUOTE)])), evidence(corpus, P2, PTA)
    )
    assert report.claims[0].verdict is ClaimVerdict.SUPPORTED
    assert len(nli.pairs) == 1  # no extra NLI work for a claim with one clause


# ------------------------------------------------------------------------ known edge cases

QUALIFIED = f"{EXEMPT_PART}, except for widgets sold to consumers."


def test_unsupported_qualifier_is_not_split_out() -> None:
    # Documented limit: a qualifier inside one clause is not a separate part. Whether the
    # claim passes is decided by the NLI model on the whole sentence alone.
    assert split_conjuncts(QUALIFIED) == []


def test_unsupported_qualifier_fails_only_if_nli_rejects_the_sentence(corpus: Corpus) -> None:
    draft = answer((QUALIFIED, [(P2, EXEMPT_QUOTE)]))
    rejecting = Verifier(corpus, RuleNLI({QUALIFIED: NEUTRAL, EXEMPT_PART: SUPPORTED}))
    report = rejecting.verify(draft, evidence(corpus, P2, PTA))
    assert report.claims[0].verdict is ClaimVerdict.UNSUPPORTED
    accepting = Verifier(corpus, RuleNLI({QUALIFIED: SUPPORTED}))
    report = accepting.verify(draft, evidence(corpus, P2, PTA))
    assert report.claims[0].verdict is ClaimVerdict.SUPPORTED  # no structural backstop
