"""Download Official Journal XHTML from the Publications Office Cellar API.

The EUR-Lex website sits behind a bot challenge; Cellar's content-negotiated resource URLs
serve the same documents to scripts. Downloads are cached under ``data/raw`` so ingestion is
reproducible offline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

CELLAR_URL = "https://publications.europa.eu/resource/celex/{celex}"
HEADERS = {
    "Accept": "application/xhtml+xml",
    "Accept-Language": "eng",
    "User-Agent": "CiteShield-ingest/0.1 (research; public EU legal texts)",
}


@dataclass(frozen=True)
class FetchedDoc:
    celex: str
    path: Path
    sha256: str
    resolved_url: str


def fetch_xhtml(
    celex: str,
    raw_dir: Path,
    *,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> FetchedDoc:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{celex}.xhtml"
    meta_path = raw_dir / f"{celex}.meta.json"

    if path.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() == meta["sha256"]:
            return FetchedDoc(celex, path, meta["sha256"], meta["resolved_url"])

    own_client = client is None
    http = client or httpx.Client(timeout=60, follow_redirects=True, headers=HEADERS)
    try:
        resp = http.get(CELLAR_URL.format(celex=celex))
        resp.raise_for_status()
    finally:
        if own_client:
            http.close()

    if "xhtml" not in resp.headers.get("content-type", ""):
        raise ValueError(f"{celex}: expected XHTML, got {resp.headers.get('content-type')!r}")

    data = resp.content
    digest = hashlib.sha256(data).hexdigest()
    path.write_bytes(data)
    meta_path.write_text(
        json.dumps(
            {
                "celex": celex,
                "sha256": digest,
                "resolved_url": str(resp.url),
                "fetched_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FetchedDoc(celex, path, digest, str(resp.url))
