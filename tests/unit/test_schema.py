from __future__ import annotations

from typing import Any

from citeshield.agent.schema import draft_schema


def _chunk_id(schema: dict[str, Any], section: str) -> dict[str, Any]:
    items = schema["properties"][section]["items"]
    chunk_id: dict[str, Any] = items["properties"]["citations"]["items"]["properties"]["chunk_id"]
    return chunk_id


def test_chunk_ids_constrained_to_evidence() -> None:
    schema = draft_schema(["b@v", "a@v", "b@v"])
    for section in ("claims", "checklist"):
        assert _chunk_id(schema, section) == {"type": "string", "enum": ["a@v", "b@v"]}


def test_no_evidence_leaves_plain_string() -> None:
    # An empty enum is invalid JSON Schema; validation rejects any citation instead.
    assert _chunk_id(draft_schema(), "claims") == {"type": "string"}


def test_schema_shape_unchanged_otherwise() -> None:
    constrained, plain = draft_schema(["a@v"]), draft_schema()
    assert constrained["required"] == plain["required"]
    assert constrained["properties"]["status"] == plain["properties"]["status"]
