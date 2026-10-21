"""Quarantine sink: rejected records are kept, with reasons, for inspection — never indexed."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from citeshield.ingest.validate import QuarantineEntry


def write_quarantine(entries: list[QuarantineEntry], path: Path, *, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            line = {"run_id": run_id, "stage": e.stage, "errors": e.errors, "record": e.record}
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")


def reason_counts(entries: list[QuarantineEntry]) -> dict[str, int]:
    """Group errors into reasons by masking the specifics (numbers, quoted values)."""
    counts: Counter[str] = Counter()
    for e in entries:
        for err in e.errors:
            reason = re.sub(r"'[^']*'|\[[^\]]*\]|\d+", "#", err)
            counts[f"{e.stage}: {reason}"] += 1
    return dict(counts.most_common())
