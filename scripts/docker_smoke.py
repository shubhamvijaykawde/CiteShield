"""Container smoke test: run the real image without index, models or LLM, and check it.

Builds the image (unless --skip-build), starts one container on a random loopback port with
the minimal configuration (Ollama provider pointed at a closed port, no index mounted) and
checks the running server:

- liveness: GET /health answers 200 with the documented structure;
- honest readiness: with no index the agent fails to load, so /health must report
  status "error" naming the missing index manifest, /ready must answer 503 and POST /ask
  must answer 503 instead of answering;
- the browser UI and the OpenAPI schema are served;
- the process runs as a non-root user, and no .env or index is baked into the image.

It downloads no models and calls no LLM, so it proves the packaging and startup path, not
answer quality (scripts/docker-e2e.ps1 does a real question). Only the container it
started is removed. Standard library only:

    python scripts/docker_smoke.py                 # build citeshield:smoke, then test
    python scripts/docker_smoke.py --image citeshield:ci --skip-build
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


class SmokeFailure(Exception):
    pass


def docker(*args: str, check: bool = True) -> str:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise SmokeFailure(f"docker {' '.join(args)} failed:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def request(
    url: str, method: str = "GET", body: dict[str, Any] | None = None
) -> tuple[int, str, str]:
    """(status, content type, body); HTTP errors are returned, connection errors raised."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ok   {message}")


def wait_for_health(base: str, timeout_s: float) -> dict[str, Any]:
    """Poll /health until the agent has stopped loading. Returns the last health JSON."""
    deadline = time.monotonic() + timeout_s
    last: str | None = None
    alive = False
    while time.monotonic() < deadline:
        try:
            code, _, text = request(f"{base}/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if code != 200:
                raise SmokeFailure(f"GET /health answered {code}: {text[:300]}")
            if not alive:
                print("  ok   GET /health answers 200 (liveness)")
                alive = True
            health: dict[str, Any] = json.loads(text)
            if health.get("status") != "loading":
                return health
            last = "status still 'loading'"
        time.sleep(2)
    raise SmokeFailure(f"no settled /health within {timeout_s:.0f} s (last: {last})")


def run_checks(base: str, container: str, timeout_s: float) -> None:
    health = wait_for_health(base, timeout_s)
    expected = {"status", "error", "llm", "busy", "tracing"}
    check(expected <= health.keys(), f"/health has keys {sorted(expected)}")
    check(
        health["status"] == "error",
        f"/health status is 'error' without an index (got {health['status']!r})",
    )
    check(
        "manifest.json" in (health["error"] or ""),
        f"/health names the missing index manifest (error: {health['error']!r})",
    )
    llm = health["llm"]
    check(
        llm.get("provider") == "ollama" and llm.get("paid") is False,
        f"/health reports the unpaid Ollama backend (llm: {llm})",
    )
    check(llm.get("reachable") is False, "/health reports the test Ollama URL as unreachable")
    check(health["busy"] is False and health["tracing"] == "off", "not busy, tracing off")

    code, _, text = request(f"{base}/ready")
    check(code == 503 and json.loads(text) == {"status": "error"}, f"GET /ready is 503 ({code})")

    code, ctype, text = request(f"{base}/ask", "POST", {"question": "What does Article 33 say?"})
    check(code == 503, f"POST /ask is refused with 503 while not ready ({code})")

    code, ctype, text = request(f"{base}/")
    check(
        code == 200 and ctype.startswith("text/html") and "CiteShield" in text,
        "GET / serves the browser UI",
    )
    code, _, text = request(f"{base}/openapi.json")
    paths = json.loads(text).get("paths", {}) if code == 200 else {}
    check({"/ask", "/health", "/ready"} <= paths.keys(), "OpenAPI lists /ask, /health, /ready")

    uid = docker("exec", container, "id", "-u")
    check(uid != "0", f"the server runs as a non-root user (uid {uid})")
    baked = docker(
        "exec", container, "sh", "-c",
        "for p in /app/.env /app/.git; do [ -e $p ] && echo $p; done; ls -A /app/data/index",
    )  # fmt: skip
    check(baked == "", "no .env, .git or index content in the image")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="citeshield:smoke")
    parser.add_argument("--skip-build", action="store_true", help="test an existing image")
    parser.add_argument("--timeout", type=float, default=180, help="startup timeout, seconds")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    try:
        docker("version", "--format", "{{.Server.Version}}")
    except (SmokeFailure, FileNotFoundError) as e:
        print(f"FAIL: no usable Docker engine: {e}", file=sys.stderr)
        return 2

    if not args.skip_build:
        print(f"Building {args.image} ...", flush=True)
        if subprocess.run(["docker", "build", "-t", args.image, "."], check=False).returncode:
            print("FAIL: docker build failed", file=sys.stderr)
            return 1

    name = f"citeshield-smoke-{uuid.uuid4().hex[:8]}"
    print(f"Starting {name} from {args.image}", flush=True)
    try:
        docker(
            "run", "--detach", "--name", name, "--label", "citeshield.test=smoke",
            "--publish", "127.0.0.1::8000",
            "--env", "CITESHIELD_LLM__PROVIDER=ollama",
            "--env", "CITESHIELD_LLM__OLLAMA_BASE_URL=http://127.0.0.1:9",
            args.image,
        )  # fmt: skip
        host_port = docker("port", name, "8000/tcp").splitlines()[0].rsplit(":", 1)[1]
        base = f"http://127.0.0.1:{host_port}"
        print(f"Checking {base}", flush=True)
        run_checks(base, name, args.timeout)
    except SmokeFailure as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        print("\n--- container logs (last 100 lines) ---", file=sys.stderr)
        logs = subprocess.run(
            ["docker", "logs", "--tail", "100", name], capture_output=True, text=True, check=False
        )
        print(logs.stdout + logs.stderr, file=sys.stderr)
        return 1
    finally:
        docker("rm", "--force", name, check=False)
    print("\nPASS: the container starts, serves the API and reports that it is not ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
