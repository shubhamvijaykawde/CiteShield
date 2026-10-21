"""JSON schemas for structured outputs.

Written by hand rather than generated from the Pydantic models: structured outputs support
a subset of JSON Schema, which differs by provider, so cross-field rules are encoded only
where the decoder can enforce them (see ``draft_schema``). Pydantic re-validates everything
the model returns.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from citeshield.contracts.answer import VERIFIER_ONLY_REASONS, AnswerStatus, RefusalReason

GUARD_LABELS = ("in_scope", "other_law", "legal_advice", "not_legal", "manipulation")


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def draft_schema(evidence_ids: Iterable[str] = (), *, status_rules: bool = False) -> dict[str, Any]:
    """Schema for the draft answer.

    ``evidence_ids`` are the chunk ids tools returned in this run. When given, a citation's
    ``chunk_id`` must be one of them, so a constrained decoder cannot emit an invented or
    mis-copied id (qwen3:4b was seen writing ``oj-2.016-05-04`` for ``oj-2016-05-04``).
    ``citation_errors`` still checks every id after parsing; the enum is an extra guard, not
    a replacement. With no evidence there is nothing citable, and the plain string type stays
    so validation can reject any citation (a JSON-schema enum cannot be empty).

    ``status_rules`` also encodes AgentAnswer's status rule for decoders that support
    ``anyOf`` and ``maxItems`` (Ollama): an answer cites something, a refusal gives a reason
    and has no claims. Anthropic's structured outputs reject ``maxItems``, so that provider
    gets the plain schema and relies on Pydantic for these rules.
    """
    ids = sorted(set(evidence_ids))
    chunk_id: dict[str, Any] = {"type": "string", "enum": ids} if ids else {"type": "string"}
    citation = _obj({"chunk_id": chunk_id, "quote": {"type": "string"}})
    citations: dict[str, Any] = {"type": "array", "items": citation}
    if status_rules:
        citations["minItems"] = 1
    claims = {
        "type": "array",
        "items": _obj({"text": {"type": "string"}, "citations": citations}),
    }
    checklist = {
        "type": "array",
        "items": _obj(
            {
                "obligation": {"type": "string"},
                "applies_to": {"type": "string"},
                "citations": citations,
            }
        ),
    }
    reason = {
        "type": "string",
        "enum": [r.value for r in RefusalReason if r not in VERIFIER_ONLY_REASONS],
    }

    optional_reason = {"anyOf": [reason, {"type": "null"}]}

    def answer(
        statuses: list[AnswerStatus], refusal_reason: dict[str, Any], **sizes: dict[str, Any]
    ) -> dict[str, Any]:
        return _obj(
            {
                "status": {"type": "string", "enum": [s.value for s in statuses]},
                "summary": {"type": "string"},
                "claims": claims | sizes.get("claims", {}),
                "checklist": checklist | sizes.get("checklist", {}),
                "refusal_reason": refusal_reason,
                "caveats": {"type": "array", "items": {"type": "string"}},
            }
        )

    if not status_rules:
        return answer(list(AnswerStatus), optional_reason)
    # One branch per way of satisfying AgentAnswer's status rule. qwen3:4b sometimes
    # returned "answered" with empty claims, and often put claims into a refusal; under
    # these branches the decoder can do neither. "answered has no refusal_reason" and quote
    # lengths stay Pydantic-only.
    content = [AnswerStatus.ANSWERED, AnswerStatus.PARTIAL]
    empty = {"maxItems": 0}
    return {
        "anyOf": [
            answer([AnswerStatus.REFUSED], reason, claims=empty, checklist=empty),
            answer(content, optional_reason, claims={"minItems": 1}),
            answer(content, optional_reason, checklist={"minItems": 1}),
        ]
    }


def guard_schema() -> dict[str, Any]:
    return _obj({"label": {"type": "string", "enum": list(GUARD_LABELS)}})
