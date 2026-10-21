"""Thin wrapper over the Anthropic SDK so the graph can run against a scripted fake in tests.

Conversation history is append-only: assistant turns are stored as the SDK's own content
blocks and sent back unchanged (thinking blocks must round-trip untouched).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

# Claude API list prices, USD per million tokens: (input, output, cache read).
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5"})
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusal(Exception):
    """The model (and its fallback chain) declined the request."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cost_usd += other.cost_usd
        self.calls += other.calls


@dataclass
class LLMResponse:
    content: list[Any]  # blocks to append verbatim as the assistant turn
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str
    model: str
    usage: Usage = field(default_factory=Usage)
    # Provider-reported metadata for the per-call trace (finish reason, lengths, token counts).
    # Never prompts, outputs or credentials.
    meta: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        return json.loads(self.text)


class LLM(Protocol):
    def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice_none: bool = False,
        json_schema: dict[str, Any] | None = None,
        effort: str = "medium",
        max_tokens: int = 16000,
    ) -> LLMResponse: ...


def estimate_cost(model: str, usage: Usage) -> float:
    price_in, price_out, price_cache = PRICES.get(model, (0.0, 0.0, 0.0))
    # Cache writes bill at 1.25x input for the 5-minute TTL.
    return (
        usage.input_tokens * price_in
        + usage.cache_write_tokens * price_in * 1.25
        + usage.cache_read_tokens * price_cache
        + usage.output_tokens * price_out
    ) / 1_000_000


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, api_key: str | None = None, *, timeout: float = 120.0) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice_none: bool = False,
        json_schema: dict[str, Any] | None = None,
        effort: str = "medium",
        max_tokens: int = 16000,
    ) -> LLMResponse:
        output_config: dict[str, Any] = {"effort": effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # The system prompt and tool list are frozen per deployment: cache the prefix.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": output_config,
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice_none:
                kwargs["tool_choice"] = {"type": "none"}
        if model in _FALLBACK_MODELS:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        resp = self.client.beta.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None)
            raise LLMRefusal(f"model declined the request (category={category})")

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            calls=1,
        )
        usage.cost_usd = estimate_cost(resp.model, usage)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            content=list(resp.content),
            text=text,
            tool_calls=[
                ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if b.type == "tool_use"
            ],
            stop_reason=resp.stop_reason or "",
            model=resp.model,
            usage=usage,
            meta={
                "provider": self.provider,
                "model": resp.model,
                "finish_reason": resp.stop_reason,
                "content_chars": len(text),
                "output_tokens": u.output_tokens,
            },
        )
