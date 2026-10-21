"""Agent configuration; its hash is logged next to prompt versions and the retriever config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_model: str = "claude-sonnet-5-5"
    guard_model: str = "claude-haiku-5-5"
    # Sonnet 5.5 guidance: start multistep tool use at medium and measure before raising.
    agent_effort: Effort = "medium"
    draft_effort: Effort = "medium"
    guard_effort: Effort = "low"
    max_tool_rounds: int = Field(6, ge=1, description="Agent turns that may call tools.")
    max_draft_attempts: int = Field(2, ge=1, description="First draft + repairs.")
    max_tokens: int = Field(16000, ge=1024)
    max_question_chars: int = Field(2000, ge=10)


Provider = Literal["anthropic", "ollama"]


class LLMSettings(BaseModel):
    """Which inference backend runs the agent. Nothing switches provider implicitly.

    With ``provider="ollama"`` every LLM call (guard, research, draft, repair) goes to the
    local Ollama server, and the Anthropic client is never constructed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider = "anthropic"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_guard_model: str | None = Field(None, description="Defaults to ollama_model.")
    # Ollama's own default (4096) silently truncates longer prompts.
    ollama_num_ctx: int = Field(16384, ge=2048)
    ollama_num_predict: int = Field(4096, ge=256, description="Output budget incl. thinking.")
    ollama_think: bool = Field(True, description="qwen3 leaks reasoning into content when off.")
    ollama_timeout_s: float = Field(600.0, gt=0)
