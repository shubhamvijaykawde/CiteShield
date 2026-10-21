"""Fixtures for opt-in live tests against local services only.

Run with: ``py -m uv run pytest -m live tests/live -s``
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}


@pytest.fixture(scope="session")
def hf_offline() -> Iterator[None]:
    """Load the cached embedding/reranker models without contacting the Hugging Face Hub.

    Must take effect before huggingface_hub is imported, which happens lazily when the
    encoder is first built inside the agent fixture.
    """
    saved = {k: os.environ.get(k) for k in _OFFLINE_ENV}
    os.environ.update(_OFFLINE_ENV)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _is_loopback(address: Any) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@pytest.fixture(scope="session")
def local_network_only() -> Iterator[list[str]]:
    """Fail any outbound connection that is not to a loopback address.

    This turns "the live test used only local inference" from a claim into a checked fact:
    a call to any hosted model API (or any other remote host) raises inside the test.
    """
    blocked: list[str] = []
    original = socket.socket.connect

    def guarded(self: socket.socket, address: Any) -> Any:
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            blocked.append(str(address))
            raise ConnectionRefusedError(f"non-local connection blocked in live test: {address}")
        return original(self, address)

    socket.socket.connect = guarded  # type: ignore[method-assign]
    try:
        yield blocked
    finally:
        socket.socket.connect = original  # type: ignore[method-assign]
