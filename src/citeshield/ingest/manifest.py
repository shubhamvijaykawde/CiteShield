"""Corpus manifest: which acts, at which pinned version."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    regulation: str = Field(pattern=r"^[A-Za-z0-9 ]{2,32}$")
    celex: str = Field(pattern=r"^[0-9]{5}[A-Z][0-9]{4}$")
    title: str
    version: str = Field(pattern=r"^[0-9A-Za-z.\-]+$")
    version_date: date
    enabled: bool = True

    @property
    def eurlex_url(self) -> str:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{self.celex}"


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[SourceSpec]


def load_manifest(path: Path) -> Manifest:
    with path.open(encoding="utf-8") as f:
        return Manifest.model_validate(yaml.safe_load(f))
