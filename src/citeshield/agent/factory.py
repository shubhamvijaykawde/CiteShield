"""Wire the production agent from settings (GPU/CPU models, Qdrant, Anthropic client)."""

from __future__ import annotations

from citeshield.agent.graph import CiteShieldAgent
from citeshield.agent.llm import AnthropicLLM
from citeshield.agent.tools import ToolExecutor
from citeshield.config import Settings
from citeshield.retrieval.pipeline import Retriever


def build_agent(settings: Settings | None = None) -> CiteShieldAgent:
    settings = settings or Settings()
    retriever = Retriever.from_settings(settings)
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return CiteShieldAgent(AnthropicLLM(api_key=key), ToolExecutor(retriever), settings.agent)
