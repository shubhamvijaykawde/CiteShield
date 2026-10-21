"""Runtime configuration.

Every pipeline run logs ``config_hash`` of the settings objects it used, so a metric in
MLflow or a trace in Langfuse can always be traced back to the exact configuration.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from citeshield.agent.config import AgentConfig
from citeshield.retrieval.config import RetrieverConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IngestConfig(BaseModel):
    """Knobs that change what ends up in the corpus."""

    max_chunk_tokens: int = Field(512, description="Soft limit; longer units are split.")
    min_chunk_tokens: int = Field(4, description="Hard lower bound enforced by validation.")
    hard_max_chunk_tokens: int = Field(1024, description="Hard upper bound enforced by validation.")
    max_quarantine_rate: float = Field(0.02, ge=0, le=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CITESHIELD_",
        env_file=PROJECT_ROOT / ".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    data_dir: Path = PROJECT_ROOT / "data"
    manifest_path: Path = PROJECT_ROOT / "data" / "corpus_manifest.yaml"

    # Read from ANTHROPIC_API_KEY (env or .env). None lets the SDK resolve credentials itself.
    anthropic_api_key: SecretStr | None = Field(None, validation_alias="ANTHROPIC_API_KEY")

    # Qdrant server URL; when unset an embedded on-disk instance under data/index is used.
    qdrant_url: str | None = None

    ingest: IngestConfig = IngestConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    agent: AgentConfig = AgentConfig()

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def quarantine_dir(self) -> Path:
        return self.data_dir / "quarantine"


def config_hash(cfg: BaseModel) -> str:
    """Stable short hash of a config object, used as a run/version identifier."""
    payload = cfg.model_dump_json(exclude_none=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
