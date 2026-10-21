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
