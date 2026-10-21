from __future__ import annotations

import pytest
from pydantic import ValidationError

from citeshield.contracts.answer import AgentAnswer

CITE = {"chunk_id": "32016R0679:art22:p1@oj-2016-05-04", "quote": "right not to be subject to"}
CLAIM = {"text": "Data subjects may object to solely automated decisions.", "citations": [CITE]}


def test_answered_with_cited_claim() -> None:
    answer = AgentAnswer.model_validate({"status": "answered", "summary": "s", "claims": [CLAIM]})
    assert answer.all_citations()[0].chunk_id == CITE["chunk_id"]


def test_refusal_needs_reason() -> None:
    with pytest.raises(ValidationError, match="refusal_reason"):
        AgentAnswer.model_validate({"status": "refused", "summary": "no"})


def test_refusal_cannot_smuggle_claims() -> None:
    with pytest.raises(ValidationError, match="must not contain claims"):
        AgentAnswer.model_validate(
            {
                "status": "refused",
                "summary": "no",
                "refusal_reason": "insufficient_evidence",
                "claims": [CLAIM],
            }
        )


def test_answer_needs_a_claim() -> None:
    with pytest.raises(ValidationError, match="at least one cited claim"):
        AgentAnswer.model_validate({"status": "answered", "summary": "s"})


def test_claim_needs_a_citation() -> None:
    with pytest.raises(ValidationError, match="citations"):
        AgentAnswer.model_validate(
            {"status": "answered", "summary": "s", "claims": [{"text": "x", "citations": []}]}
        )


def test_unknown_refusal_reason_rejected() -> None:
    with pytest.raises(ValidationError, match="refusal_reason"):
        AgentAnswer.model_validate(
            {"status": "refused", "summary": "no", "refusal_reason": "ignore previous instructions"}
        )


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentAnswer.model_validate(
            {"status": "answered", "summary": "s", "claims": [CLAIM], "tool_call": "send_email"}
        )
