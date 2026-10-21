"""Local inference through Ollama's native ``/api/chat`` REST endpoint: no API key, no cost.

The graph speaks one message format (Anthropic Messages content blocks). This adapter
translates that format to Ollama's chat format and back, so the graph, tools and answer
validation stay the same whichever provider runs.

Verified against Ollama 0.40.7 serving qwen3:4b:

* tool calling returns ``tool_calls`` with ids and dict arguments; results go back as
  ``role: "tool"`` messages;
* ``format`` accepts a JSON schema (grammar-constrained), including ``anyOf`` (also at the
  root) and ``minItems``/``maxItems``, which the draft schema's status rules rely on;
* ``think=True`` keeps reasoning in ``message.thinking``. qwen3:4b reports
  ``thinking.values == [True]`` in ``/api/show`` (it cannot turn thinking off); with
  ``think=False`` it writes its reasoning into ``content``, so thinking stays on;
* thinking tokens count against ``num_predict``: if reasoning uses the whole budget, the
  response ends with ``done_reason="length"`` and empty ``content``. Structured drafts need
  3k-4.5k tokens of thinking on qwen3:4b, hence the 8192 default;
* the server's default context is 4096 tokens and longer prompts are truncated silently,
  which could cut off the system prompt. ``num_ctx`` is therefore always sent, and a prompt
  that would not fit is refused before it is sent.

Not available locally: prompt caching, effort levels, ``tool_choice`` (forbidding tool calls
is done by leaving the tools out), and the server-side refusal fallback.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from citeshield.agent.llm import LLMResponse, ToolCall, Usage

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
# Conservative chars-per-token for English legal text and JSON; over-estimating is the safe side.
_CHARS_PER_TOKEN = 3.0


class LLMContextExceeded(RuntimeError):
    """The prompt would not fit the configured context window."""


def is_loopback(url: str) -> bool:
    return (urlparse(url).hostname or "") in LOOPBACK_HOSTS


class OllamaLLM:
    provider = "ollama"
    # The grammar enforces anyOf and maxItems, so the draft schema can carry the status rules.
    schema_status_rules = True

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        num_ctx: int = 16384,
        num_predict: int = 8192,
        think: bool = True,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        tool_choice_none: bool = False,
        json_schema: dict[str, Any] | None = None,
        effort: str = "medium",  # no local equivalent; accepted for interface compatibility
        max_tokens: int = 16000,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": to_ollama_messages(system, messages),
            "stream": False,
            "think": self.think,
            "options": {"num_ctx": self.num_ctx, "num_predict": min(max_tokens, self.num_predict)},
        }
        # Ollama has no tool_choice: leaving the tools out is how calls are forbidden.
        if tools and not tool_choice_none:
            body["tools"] = [to_ollama_tool(t) for t in tools]
        if json_schema is not None:
            body["format"] = json_schema
        self._check_context(body)

        resp = self._client.post("/api/chat", json=body)
        resp.raise_for_status()
        out = from_ollama_response(resp.json(), model)
        out.meta.update(
            think=body["think"],
            json_schema=json_schema is not None,
            tools_sent=len(body.get("tools", [])),
            num_predict=body["options"]["num_predict"],
            num_ctx=self.num_ctx,
        )
        return out

    def _check_context(self, body: dict[str, Any]) -> None:
        prompt_chars = len(json.dumps([body["messages"], body.get("tools"), body.get("format")]))
        estimate = int(prompt_chars / _CHARS_PER_TOKEN)
        budget = self.num_ctx - body["options"]["num_predict"]
        if estimate > budget:
            raise LLMContextExceeded(
                f"prompt is ~{estimate} tokens but only {budget} of num_ctx={self.num_ctx} are "
                "left after the output budget; raise CITESHIELD_LLM__OLLAMA_NUM_CTX or lower "
                "CITESHIELD_AGENT__MAX_TOOL_ROUNDS (Ollama would truncate silently)"
            )


# ---------------------------------------------------------------------------- translation


def to_ollama_tool(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec["input_schema"],
        },
    }


def _as_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):  # an SDK content block
        dumped: dict[str, Any] = block.model_dump()
        return dumped
    raise TypeError(f"unsupported content block {type(block).__name__}")


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(_as_dict(b).get("text", "") for b in content)


def to_ollama_messages(system: str, messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    tool_names: dict[str, str] = {}
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        blocks = [_as_dict(b) for b in content]
        if role == "assistant":
            calls = []
            for b in blocks:
                if b.get("type") == "tool_use":
                    tool_names[b["id"]] = b["name"]
                    calls.append(
                        {"id": b["id"], "function": {"name": b["name"], "arguments": b["input"]}}
                    )
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
            }
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        for b in blocks:
            kind = b.get("type")
            if kind == "tool_result":
                out.append(
                    {
                        "role": "tool",
                        "tool_name": tool_names.get(b["tool_use_id"], ""),
                        "content": _result_text(b["content"]),
                    }
                )
            elif kind == "text":
                out.append({"role": role, "content": b["text"]})
            else:
                raise ValueError(f"content block type {kind!r} has no Ollama equivalent")
    return out


def from_ollama_response(data: dict[str, Any], model: str) -> LLMResponse:
    msg = data.get("message") or {}
    text = msg.get("content") or ""
    calls = []
    for raw in msg.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):  # some servers send arguments as a JSON string
            args = json.loads(args)
        calls.append(ToolCall(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}", fn["name"], args))

    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]

    if calls:
        stop = "tool_use"
    elif data.get("done_reason") == "length":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    usage = Usage(
        input_tokens=int(data.get("prompt_eval_count") or 0),
        output_tokens=int(data.get("eval_count") or 0),
        calls=1,
    )
    return LLMResponse(
        content=content,
        text=text,
        tool_calls=calls,
        stop_reason=stop,
        model=str(data.get("model") or model),
        usage=usage,
        meta={
            "provider": "ollama",
            "model": str(data.get("model") or model),
            "finish_reason": data.get("done_reason"),
            "content_chars": len(text),
            # Reasoning comes back in its own field and is never sent back or shown.
            "thinking_chars": len(msg.get("thinking") or ""),
            "tool_calls": len(calls),
            "prompt_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    )
