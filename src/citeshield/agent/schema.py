"""JSON schemas for structured outputs.

Written by hand rather than generated from the Pydantic models: structured outputs support
a subset of JSON Schema, and the cross-field rules (a refusal has no claims, ...) cannot be
expressed in it anyway. Pydantic re-validates everything the model returns.
"""

from __future__ import annotations

from typing import Any

from citeshield.contracts.answer import AnswerStatus, RefusalReason

GUARD_LABELS = ("in_scope", "other_law", "legal_advice", "not_legal", "manipulation")


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def draft_schema() -> dict[str, Any]:
    citation = _obj({"chunk_id": {"type": "string"}, "quote": {"type": "string"}})
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
                    {"type": "string", "enum": [r.value for r in RefusalReason]},
                    {"type": "null"},
                ]
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        }
    )


def guard_schema() -> dict[str, Any]:
    return _obj({"label": {"type": "string", "enum": list(GUARD_LABELS)}})
