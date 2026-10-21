"""Wire the production agent from settings (GPU/CPU models, Qdrant, LLM provider)."""

from __future__ import annotations

from citeshield.agent.config import AgentConfig
from citeshield.agent.graph import CiteShieldAgent
from citeshield.agent.llm import LLM, AnthropicLLM
from citeshield.agent.tools import ToolExecutor
from citeshield.config import Settings
from citeshield.retrieval.corpus import Corpus
from citeshield.retrieval.pipeline import Retriever
from citeshield.verify.verifier import Verifier


def build_llm(settings: Settings) -> tuple[LLM, AgentConfig]:
    """The configured LLM client and the agent config with that provider's model names."""
    llm = settings.llm
    if llm.provider == "ollama":
        from citeshield.agent.ollama import OllamaLLM

        client = OllamaLLM(
            llm.ollama_base_url,
            num_ctx=llm.ollama_num_ctx,
            num_predict=llm.ollama_num_predict,
            think=llm.ollama_think,
            timeout=llm.ollama_timeout_s,
        )
        cfg = settings.agent.model_copy(
            update={
                "agent_model": llm.ollama_model,
                "guard_model": llm.ollama_guard_model or llm.ollama_model,
            }
        )
        return client, cfg
    if llm.provider == "anthropic":
        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicLLM(api_key=key), settings.agent
    raise ValueError(f"unknown LLM provider {llm.provider!r}")


def build_verifier(settings: Settings, corpus: Corpus) -> Verifier:
    from citeshield.verify.nli import DebertaNLI

    cfg = settings.verifier
    nli = DebertaNLI(
        cfg.nli_model, revision=cfg.nli_revision, device=cfg.device, batch_size=cfg.batch_size
    )
    return Verifier(corpus, nli, cfg)


def build_agent(settings: Settings | None = None) -> CiteShieldAgent:
    settings = settings or Settings()
    llm, cfg = build_llm(settings)
    retriever = Retriever.from_settings(settings)
    tools = ToolExecutor(retriever)
    return CiteShieldAgent(llm, tools, cfg, verifier=build_verifier(settings, tools.corpus))
