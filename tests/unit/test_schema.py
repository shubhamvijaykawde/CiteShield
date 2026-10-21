from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from citeshield.agent.schema import draft_schema
from citeshield.contracts.answer import AgentAnswer


def _objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """The answer object(s): the plain schema itself, or each status-rules branch."""
    branches: list[dict[str, Any]] = schema.get("anyOf", [schema])
    return branches


def _chunk_ids(schema: dict[str, Any], section: str) -> list[dict[str, Any]]:
    return [
        b["properties"][section]["items"]["properties"]["citations"]["items"]["properties"][
            "chunk_id"
        ]
        for b in _objects(schema)
    ]


@pytest.mark.parametrize("status_rules", [False, True])
def test_chunk_ids_constrained_to_evidence(status_rules: bool) -> None:
    schema = draft_schema(["b@v", "a@v", "b@v"], status_rules=status_rules)
    for section in ("claims", "checklist"):
        for chunk_id in _chunk_ids(schema, section):
            assert chunk_id == {"type": "string", "enum": ["a@v", "b@v"]}


@pytest.mark.parametrize("status_rules", [False, True])
def test_no_evidence_leaves_plain_string(status_rules: bool) -> None:
    # An empty enum is invalid JSON Schema; validation rejects any citation instead.
    schema = draft_schema(status_rules=status_rules)
    assert all(c == {"type": "string"} for c in _chunk_ids(schema, "claims"))


@pytest.mark.parametrize("status_rules", [False, True])
def test_schema_shape_unchanged_otherwise(status_rules: bool) -> None:
    constrained = draft_schema(["a@v"], status_rules=status_rules)
    plain = draft_schema(status_rules=status_rules)
    for c, p in zip(_objects(constrained), _objects(plain), strict=True):
        assert c["required"] == p["required"]
        assert c["properties"]["status"] == p["properties"]["status"]


# ------------------------------------------------------- schema vs AgentAnswer contract

_CITE = {"chunk_id": "a@v", "quote": "shall not apply to widgets"}
_CLAIM = {"text": "Widgets are exempt.", "citations": [_CITE]}
_ITEM = {"obligation": "Notify", "applies_to": "Providers", "citations": [_CITE]}


def _draft(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "answered",
        "summary": "Widgets used for research are exempt.",
        "claims": [_CLAIM],
        "checklist": [],
        "refusal_reason": None,
        "caveats": [],
    }
    return base | overrides


def _schema_ok(draft: dict[str, Any], *, status_rules: bool = True) -> bool:
    schema = draft_schema(["a@v"], status_rules=status_rules)
    return jsonschema.Draft202012Validator(schema).is_valid(draft)


def _model_ok(draft: dict[str, Any]) -> bool:
    try:
        AgentAnswer.model_validate({**draft, "checklist": draft["checklist"] or None})
    except ValidationError:
        return False
    return True


# Rules the status-rules schema enforces: it and AgentAnswer must agree on every case.
_ENFORCED = {
    "answered with a claim": (_draft(), True),
    "answered with a checklist only": (_draft(claims=[], checklist=[_ITEM]), True),
    "partial with a reason": (
        _draft(status="partial", refusal_reason="insufficient_evidence"),
        True,
    ),
    "refused with a reason": (
        _draft(status="refused", claims=[], refusal_reason="out_of_scope"),
        True,
    ),
    "answered with nothing cited": (_draft(claims=[]), False),
    "partial with nothing cited": (
        _draft(status="partial", claims=[], refusal_reason="insufficient_evidence"),
        False,
    ),
    "refused without a reason": (_draft(status="refused", claims=[]), False),
    "refused with claims": (_draft(status="refused", refusal_reason="out_of_scope"), False),
    "refused with a checklist": (
        _draft(status="refused", claims=[], checklist=[_ITEM], refusal_reason="out_of_scope"),
        False,
    ),
    "claim without a citation": (_draft(claims=[{"text": "x", "citations": []}]), False),
    "checklist item without a citation": (
        _draft(claims=[], checklist=[{**_ITEM, "citations": []}]),
        False,
    ),
}


@pytest.mark.parametrize("case", list(_ENFORCED))
def test_status_rules_schema_enforces_answer_contract(case: str) -> None:
    draft, ok = _ENFORCED[case]
    assert _schema_ok(draft) is ok
    assert _model_ok(draft) is ok


@pytest.mark.parametrize("case", [c for c, (_, ok) in _ENFORCED.items() if not ok])
def test_plain_schema_leaves_status_rules_to_pydantic(case: str) -> None:
    draft, _ = _ENFORCED[case]
    assert _schema_ok(draft, status_rules=False)
    assert not _model_ok(draft)


# Rules no schema carries here: the decoder lets these through, AgentAnswer rejects them.
_PYDANTIC_ONLY = {
    "answered with a refusal reason": _draft(refusal_reason="insufficient_evidence"),
    "quote shorter than 8 characters": _draft(
        claims=[{"text": "x", "citations": [{"chunk_id": "a@v", "quote": "short"}]}]
    ),
}


@pytest.mark.parametrize("case", list(_PYDANTIC_ONLY))
def test_rules_left_to_pydantic(case: str) -> None:
    draft = _PYDANTIC_ONLY[case]
    assert _schema_ok(draft)
    assert not _model_ok(draft)


@pytest.mark.parametrize("status_rules", [False, True])
def test_verifier_only_reasons_not_offered_to_the_model(status_rules: bool) -> None:
    draft = _draft(status="refused", claims=[], refusal_reason="unsupported_claims")
    assert not _schema_ok(draft, status_rules=status_rules)
    assert _model_ok(draft)  # valid when the verifier sets it


# ------------------------------------------------- Anthropic structured-output subset


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_plain_schema_stays_in_anthropic_subset() -> None:
    # The schema Anthropic receives (platform.claude.com structured outputs): minItems only
    # 0 or 1, no maxItems or string lengths, objects closed with every property required.
    nodes = list(_walk(draft_schema(["a@v"])))
    assert not any({"maxItems", "minLength", "maxLength", "oneOf", "not"} & n.keys() for n in nodes)
    assert all(n["minItems"] in (0, 1) for n in nodes if "minItems" in n)
    for n in nodes:
        if n.get("type") == "object":
            assert n["additionalProperties"] is False
            assert n["required"] == list(n["properties"])
