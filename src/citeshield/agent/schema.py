"""JSON schemas for structured outputs.

Written by hand rather than generated from the Pydantic models: structured outputs support
a subset of JSON Schema, and the cross-field rules (a refusal has no claims, ...) cannot be
expressed in it anyway. Pydantic re-validates everything the model returns.
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


def draft_schema(evidence_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Schema for the draft answer.

    ``evidence_ids`` are the chunk ids tools returned in this run. When given, a citation's
    ``chunk_id`` must be one of them, so a constrained decoder cannot emit an invented or
    mis-copied id (qwen3:4b was seen writing ``oj-2.016-05-04`` for ``oj-2016-05-04``).
    ``citation_errors`` still checks every id after parsing; the enum is an extra guard, not
    a replacement. With no evidence there is nothing citable, and the plain string type stays
    so validation can reject any citation (a JSON-schema enum cannot be empty).
    """
    ids = sorted(set(evidence_ids))
    chunk_id: dict[str, Any] = {"type": "string", "enum": ids} if ids else {"type": "string"}
    citation = _obj({"chunk_id": chunk_id, "quote": {"type": "string"}})
    citations = {"type": "array", "items": citation}
    return _obj(
        {
            "status": {"type": "string", "enum": [s.value for s in AnswerStatus]},
            "summary": {"type": "string"},
            "claims": {
                "type": "array",
                "items": _obj({"text": {"type": "string"}, "citations": citations}),
            },
            "checklist": {
                "type": "array",
                "items": _obj(
                    {
                        "obligation": {"type": "string"},
                        "applies_to": {"type": "string"},
                        "citations": citations,
                    }
                ),
            },
            "refusal_reason": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": [r.value for r in RefusalReason if r not in VERIFIER_ONLY_REASONS],
                    },
                    {"type": "null"},
                ]
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        }
    )


def guard_schema() -> dict[str, Any]:
    return _obj({"label": {"type": "string", "enum": list(GUARD_LABELS)}})
