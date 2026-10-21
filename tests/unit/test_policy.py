"""Refusal policy: every outcome, and the rule that verification never raises a status."""

from __future__ import annotations

import pytest

from citeshield.agent.policy import DISCLAIMER, finalize
from citeshield.contracts.answer import AgentAnswer, AnswerStatus, RefusalReason
from citeshield.retrieval.corpus import Corpus
from citeshield.verify.verifier import Verifier
from helpers import RuleNLI
from vfx import (
    EXEMPT,
    EXEMPT_QUOTE,
    NEUTRAL,
    P2,
    PTA,
    SUPPORTED,
    answer,
    evidence,
)

MIN = 0.30
RECORDS = "Providers must keep records of every widget test for ten years."
RECORDS_QUOTE = "keep records of every widget test for a period of ten years"


def run(corpus: Corpus, draft: AgentAnswer, nli: RuleNLI, score: float | None = 0.9) -> AgentAnswer:
    ev = evidence(corpus, P2, PTA, score=score)
    return finalize(draft, Verifier(corpus, nli).verify(draft, ev), MIN)


def test_fully_supported_answer_is_accepted(corpus: Corpus) -> None:
    nli = RuleNLI({EXEMPT: SUPPORTED})
    final = run(corpus, answer((EXEMPT, [(P2, EXEMPT_QUOTE)]), summary=EXEMPT), nli)
    assert final.status is AnswerStatus.ANSWERED and final.refusal_reason is None
    assert final.summary == EXEMPT
    assert final.caveats[-1] == DISCLAIMER


def test_mixed_claims_become_partial_with_failed_removed(corpus: Corpus) -> None:
    nli = RuleNLI({EXEMPT: SUPPORTED, RECORDS: NEUTRAL})
    draft = answer((EXEMPT, [(P2, EXEMPT_QUOTE)]), (RECORDS, [(PTA, RECORDS_QUOTE)]))
    final = run(corpus, draft, nli)
    assert final.status is AnswerStatus.PARTIAL
    assert final.refusal_reason is RefusalReason.UNSUPPORTED_CLAIMS
    assert [c.text for c in final.claims] == [EXEMPT]
    assert any("Removed 1 statement" in c for c in final.caveats)
    # An unverifiable summary is rebuilt from verified claims, never kept.
    assert final.summary == EXEMPT


def test_failed_citation_is_stripped_from_a_supported_claim(corpus: Corpus) -> None:
    nli = RuleNLI({EXEMPT: SUPPORTED})
    draft = answer((EXEMPT, [(P2, EXEMPT_QUOTE), (P2.replace("p2", "p7"), EXEMPT_QUOTE)]))
    final = run(corpus, draft, nli)
    assert [c.chunk_id for c in final.claims[0].citations] == [P2]


@pytest.mark.parametrize(
    ("claims", "nli", "reason"),
    [
        # every claim fails on its citations
        ([(EXEMPT, [(P2, "a quote that is not in the text")])],
         RuleNLI({}),
         RefusalReason.INVALID_CITATIONS),
        # citations fine, but the text does not say it
        ([(RECORDS, [(P2, EXEMPT_QUOTE)])],
         RuleNLI({RECORDS: NEUTRAL}),
         RefusalReason.UNSUPPORTED_CLAIMS),
        # sources disagree
        ([(EXEMPT, [(P2, EXEMPT_QUOTE), (PTA, RECORDS_QUOTE)])],
         RuleNLI({EXEMPT: SUPPORTED}, contradicting_premise="ten years"),
         RefusalReason.CONFLICTING_SOURCES),
    ],
)  # fmt: skip
def test_nothing_verified_refuses_with_specific_reason(
    corpus: Corpus,
    claims: list,
    nli: RuleNLI,
    reason: RefusalReason,  # type: ignore[type-arg]
) -> None:
    final = run(corpus, answer(*claims, summary="Some confident statement."), nli)
    assert final.status is AnswerStatus.REFUSED
    assert final.refusal_reason is reason
    assert final.claims == [] and "confident" not in final.summary


def test_weak_retrieval_evidence_refuses(corpus: Corpus) -> None:
    final = run(corpus, answer((EXEMPT, [(P2, EXEMPT_QUOTE)])), RuleNLI({EXEMPT: SUPPORTED}), 0.1)
    assert final.status is AnswerStatus.REFUSED
    assert final.refusal_reason is RefusalReason.INSUFFICIENT_EVIDENCE


def test_drafted_refusal_keeps_reason_and_drops_model_prose(corpus: Corpus) -> None:
    draft = AgentAnswer.model_validate(
        {
            "status": "refused",
            "summary": "Not covered, but the CRA requires 24h reporting.",  # unverifiable claim
            "refusal_reason": "out_of_scope",
            "caveats": ["The CRA applies from 2027."],
        }
    )
    final = run(corpus, draft, RuleNLI({}))
    assert final.status is AnswerStatus.REFUSED
    assert final.refusal_reason is RefusalReason.OUT_OF_SCOPE
    assert "CRA" not in final.summary and final.caveats == []


@pytest.mark.parametrize("draft_status", ["answered", "partial"])
@pytest.mark.parametrize("entail", [SUPPORTED, NEUTRAL])
def test_verification_never_raises_status(
    corpus: Corpus, draft_status: str, entail: tuple[float, float, float]
) -> None:
    rank = {AnswerStatus.REFUSED: 0, AnswerStatus.PARTIAL: 1, AnswerStatus.ANSWERED: 2}
    draft = answer((EXEMPT, [(P2, EXEMPT_QUOTE)]), status=draft_status)
    final = run(corpus, draft, RuleNLI({EXEMPT: entail}))
    assert rank[final.status] <= rank[draft.status]
