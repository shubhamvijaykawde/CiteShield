"""Versioned prompts. A prompt's version is the hash of its file, logged with every run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def version(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:8]


@cache
def load(name: str) -> Prompt:
    return Prompt(name, (_DIR / f"{name}.md").read_text(encoding="utf-8"))


def versions() -> dict[str, str]:
    return {p.stem: load(p.stem).version for p in sorted(_DIR.glob("*.md"))}
