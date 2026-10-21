"""Refusal policy: turn a drafted answer plus its verification report into the final answer.

Rules (each one has a test in tests/unit/test_policy.py):

* the guard's out-of-scope / advice / manipulation refusals happen before research;
* weak retrieval evidence (best binding-source rerank score below ``min_retrieval_score``)
  refuses with ``insufficient_evidence`` before any claim is considered;
* a drafted refusal stays a refusal, with a fixed explanation (the model's prose is dropped);
* claims and checklist items that fail verification are removed, and so are citations that
  failed their own checks; if nothing is left, the answer is refused with the most specific
  reason: ``conflicting_sources`` > ``invalid_citations`` (every failure was a citation
  failure) > ``unsupported_claims``;
* if anything was removed the answer is at most ``partial``;
* only summary and caveat sentences that passed verification are kept; an empty summary is
  rebuilt from the verified claims; fixed disclaimers are added by code.

Verification can only keep or lower an answer's status, never raise it.
"""

from __future__ import annotations

from citeshield.contracts.answer import (
    AgentAnswer,
    AnswerStatus,
    ChecklistItem,
    Citation,
    Claim,
    RefusalReason,
)
from citeshield.contracts.verification import ClaimCheck, ClaimVerdict, VerificationReport

DISCLAIMER = "General information from the EU legal texts, not legal advice."

REFUSAL_SUMMARIES: dict[RefusalReason, str] = {
    RefusalReason.INSUFFICIENT_EVIDENCE: (
        "I could not find provisions in the EU texts I cover that answer this question."
    ),
    RefusalReason.OUT_OF_SCOPE: (
        "This question is about law or facts outside the EU texts I cover "
        "(GDPR, AI Act, DSA, NIS2, Data Act), so I can't answer it from my sources."
    ),
    RefusalReason.LEGAL_ADVICE_REQUEST: (
        "I can't assess your specific situation or confirm compliance. I can explain what the "
        "regulations require of a given type of organisation or system, with citations."
    ),
    RefusalReason.CONFLICTING_SOURCES: (
        "The provisions I retrieved point in different directions on this question, so I "
        "won't give a single answer."
    ),
    RefusalReason.INJECTION_DETECTED: (
        "This request tries to change how I work, so I won't act on it."
    ),
    RefusalReason.INVALID_CITATIONS: (
        "I could not produce an answer whose citations check out against the official texts."
    ),
    RefusalReason.UNSUPPORTED_CLAIMS: (
        "The provisions I retrieved do not support the statements needed to answer this question."
    ),
}

_RANK = {AnswerStatus.REFUSED: 0, AnswerStatus.PARTIAL: 1, AnswerStatus.ANSWERED: 2}


def refusal(reason: RefusalReason, caveats: list[str] | None = None) -> AgentAnswer:
    return AgentAnswer(
        status=AnswerStatus.REFUSED,
        summary=REFUSAL_SUMMARIES[reason],
        refusal_reason=reason,
        caveats=caveats or [],
    )


def weak_evidence(report: VerificationReport, min_score: float) -> bool:
    top = report.retrieval_top_score
    return top is not None and top < min_score


def failure_reason(failed: list[ClaimCheck]) -> RefusalReason:
    verdicts = {c.verdict for c in failed}
    if ClaimVerdict.CONFLICTING in verdicts:
        return RefusalReason.CONFLICTING_SOURCES
    if verdicts == {ClaimVerdict.CITATION_INVALID}:
        return RefusalReason.INVALID_CITATIONS
    return RefusalReason.UNSUPPORTED_CLAIMS


def revision_feedback(report: VerificationReport) -> str:
    lines = []
    for c in report.failed:
        label = "Claim" if c.kind == "claim" else "Checklist item"
        lines.append(f"- {label} {c.index + 1} ({c.verdict.value}): {c.text!r}")
        lines += [f"    * {r}" for r in c.reasons[:4]]
    return (
        "Verification against the official texts failed for these statements:\n"
        + "\n".join(lines)
        + "\n\nReturn the full corrected JSON. Keep only statements that the cited text "
        "itself states; copy quotes character for character from the <source> text; cite only "
        "ids returned by tools. Remove any statement you cannot support. If nothing can be "
        "supported, refuse with refusal_reason insufficient_evidence."
    )


def finalize(
    draft: AgentAnswer, report: VerificationReport, min_retrieval_score: float
) -> AgentAnswer:
    if draft.status is AnswerStatus.REFUSED:
        return refusal(draft.refusal_reason or RefusalReason.INSUFFICIENT_EVIDENCE)
    if weak_evidence(report, min_retrieval_score):
        return refusal(
            RefusalReason.INSUFFICIENT_EVIDENCE,
            [
                f"Best retrieval score {report.retrieval_top_score:.2f} is below "
                f"{min_retrieval_score:.2f}."
            ],
        )

    kept_claims: list[Claim] = []
    kept_items: list[ChecklistItem] = []
    for check in report.supported:
        if check.kind == "claim":
            claim = draft.claims[check.index]
            kept_claims.append(
                claim.model_copy(update={"citations": _usable(claim.citations, check)})
            )
        else:
            item = (draft.checklist or [])[check.index]
            kept_items.append(item.model_copy(update={"citations": _usable(item.citations, check)}))

    failed = report.failed
    if not kept_claims and not kept_items:
        why = failure_reason(failed) if failed else RefusalReason.INSUFFICIENT_EVIDENCE
        return refusal(why, _removal_notes(failed))

    status: AnswerStatus = draft.status
    reason: RefusalReason | None = draft.refusal_reason
    if failed:
        status, reason = AnswerStatus.PARTIAL, failure_reason(failed)

    summary = " ".join(s.text for s in report.sentences if s.source == "summary" and s.kept)
    if not summary:
        summary = " ".join(c.text for c in kept_claims) or " ".join(
            i.obligation for i in kept_items
        )
    caveats = [s.text for s in report.sentences if s.source == "caveat" and s.kept]
    caveats += _removal_notes(failed)
    caveats.append(DISCLAIMER)

    final = AgentAnswer(
        status=status,
        summary=summary[:2000],
        claims=kept_claims,
        checklist=kept_items or None,
        refusal_reason=reason,
        caveats=caveats,
        corpus_version=draft.corpus_version,
    )
    if _RANK[final.status] > _RANK[draft.status]:  # pragma: no cover - defensive invariant
        raise AssertionError("verification must never raise an answer's status")
    return final


def _usable(citations: list[Citation], check: ClaimCheck) -> list[Citation]:
    """Keep only the citations that passed their own checks; failed ones are not shown."""
    ok = {c.chunk_id for c in check.citations if c.usable}
    return [c for c in citations if c.chunk_id in ok]


def _removal_notes(failed: list[ClaimCheck]) -> list[str]:
    if not failed:
        return []
    return [
        f"Removed {len(failed)} statement(s) that did not pass verification against the "
        "cited texts."
    ]
