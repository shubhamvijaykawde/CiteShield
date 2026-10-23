"""FastAPI app: the local browser demo (``/``) and the API it uses.

``POST /ask`` streams the agent's progress as Server-Sent Events:

    event: status       {"stage": "guard" | "research" | "draft", ...}
    event: guard        {"label": "in_scope" | ...}
    event: tool_call    {"name": "retrieve", "input": {...}}
    event: tool_result  {"name": ..., "n_chunks": 8, "sources": [{chunk_id, label, score}]}
    event: note         {"text": "..."}             research notes between tool calls
    event: draft_rejected {"errors": [...]}         validation sent the draft back
    event: final        {"answer": AgentAnswer, "sources": {...}, "verification": {...}, ...}
    event: error        {"message": "..."}

Other endpoints: ``GET /health`` (always 200: status loading / ok / error, the LLM
backend, startup diagnostics), ``GET /ready`` (200 once the agent is loaded, else 503),
``GET /corpus``, ``GET /sources/{chunk_id}``.

The agent (encoder, reranker, NLI model, index) loads in the background, so the page and
``/health`` answer while it loads. One question runs at a time (a second gets 429): local
models use most of a laptop's memory. Limits: request body ``CITESHIELD_API_MAX_BODY``
bytes (default 16384), answer time ``CITESHIELD_API_TIMEOUT_S`` seconds (default 900).

Run: ``uv run uvicorn citeshield.api.main:app`` (binds 127.0.0.1:8000).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from citeshield.agent.graph import CiteShieldAgent, Mode
from citeshield.observability import langfuse_configured, traced

log = logging.getLogger("citeshield.api")

AgentFactory = Callable[[], CiteShieldAgent]
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    regulations: list[str] | None = Field(None, max_length=5)
    mode: Mode = "answer"


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def llm_diagnostics() -> dict[str, Any]:
    """Which LLM backend the agent will call, and whether it looks usable. Never raises."""
    try:
        from citeshield.config import Settings

        llm = Settings().llm
        if llm.provider == "ollama":
            import httpx

            info: dict[str, Any] = {
                "provider": "ollama",
                "model": llm.ollama_model,
                "base_url": llm.ollama_base_url,
                "paid": False,
            }
            try:
                tags = httpx.get(f"{llm.ollama_base_url}/api/tags", timeout=3).json()
                names = {m.get("name") for m in tags.get("models", [])}
                info |= {"reachable": True, "model_present": llm.ollama_model in names}
            except (httpx.HTTPError, ValueError):
                info |= {"reachable": False, "model_present": False}
            return info
        settings = Settings()
        return {
            "provider": llm.provider,
            "paid": True,
            "api_key_configured": settings.anthropic_api_key is not None
            or bool(os.environ.get("ANTHROPIC_API_KEY")),
        }
    except Exception as e:
        return {"provider": "unknown", "error": f"{type(e).__name__}: {e}"}


def create_app(
    agent_factory: AgentFactory | None = None,
    *,
    load_in_background: bool | None = None,
    diagnostics: Callable[[], dict[str, Any]] | None = None,
) -> FastAPI:
    """``agent_factory=None`` builds the production agent from settings, in the background
    by default; with a factory (tests) it loads before the first request unless told
    otherwise."""
    background = agent_factory is None if load_in_background is None else load_in_background
    max_body = int(os.environ.get("CITESHIELD_API_MAX_BODY", "16384"))
    timeout_s = float(os.environ.get("CITESHIELD_API_TIMEOUT_S", "900"))

    async def load(app: FastAPI) -> None:
        if agent_factory is None:
            from citeshield.agent.factory import build_agent

            factory: AgentFactory = build_agent
        else:
            factory = agent_factory
        t0 = time.perf_counter()
        try:
            app.state.agent = await asyncio.to_thread(factory)
            app.state.status = "ok"
            log.info("agent ready in %.1fs", time.perf_counter() - t0)
        except Exception as e:
            app.state.status, app.state.error = "error", f"{type(e).__name__}: {e}"
            log.exception("agent failed to load; see GET /health")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.status, app.state.error, app.state.agent = "loading", None, None
        app.state.busy = threading.Lock()
        app.state.llm = (diagnostics or llm_diagnostics)()
        log.info("LLM backend: %s", app.state.llm)
        task = asyncio.create_task(load(app)) if background else None
        if task is None:
            await load(app)
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(title="CiteShield", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        size = request.headers.get("content-length")
        if size is not None and size.isdigit() and int(size) > max_body:
            return JSONResponse({"detail": f"request body over {max_body} bytes"}, 413)
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Revalidate the UI files on every load (cheap 304s), so an updated UI is never
            # shown with a stale script or stylesheet from the browser cache.
            response.headers["Cache-Control"] = "no-cache"
        return response

    def get_agent(request: Request) -> CiteShieldAgent:
        state = request.app.state
        if state.status != "ok":
            detail = state.error if state.status == "error" else "the agent is still loading"
            raise HTTPException(503, detail)
        agent: CiteShieldAgent = state.agent
        return agent

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        out: dict[str, Any] = {
            "status": state.status,
            "error": state.error,
            "llm": state.llm,
            "busy": state.busy.locked(),
            "tracing": "langfuse" if langfuse_configured() else "off",
        }
        if state.status == "ok":
            out |= state.agent.meta
            out["regulations"] = state.agent.tools.regulations
        return out

    @app.get("/ready")
    def ready(request: Request) -> JSONResponse:
        status = request.app.state.status
        return JSONResponse({"status": status}, 200 if status == "ok" else 503)

    @app.get("/corpus")
    def corpus(request: Request) -> dict[str, Any]:
        agent = get_agent(request)
        counts: dict[str, int] = {}
        for c in agent.tools.corpus:
            counts[c.regulation] = counts.get(c.regulation, 0) + 1
        return {"corpus_version": agent.corpus_version, "chunks_by_regulation": counts}

    @app.get("/sources/{chunk_id:path}")
    def source(chunk_id: str, request: Request) -> dict[str, Any]:
        chunk = get_agent(request).tools.corpus.get(chunk_id)
        if chunk is None:
            raise HTTPException(404, "unknown chunk_id")
        return {
            "chunk_id": chunk.chunk_id,
            "label": chunk.citation_label,
            "regulation": chunk.regulation,
            "celex": chunk.celex,
            "version": chunk.version,
            "heading_path": chunk.heading_path,
            "text": chunk.text,
            "url": str(chunk.source_url),
        }

    @app.post("/ask")
    async def ask(body: AskRequest, request: Request) -> StreamingResponse:
        agent = get_agent(request)
        unknown = set(body.regulations or []) - set(agent.tools.regulations)
        if unknown:
            raise HTTPException(422, f"unknown regulations: {sorted(unknown)}")
        busy: threading.Lock = request.app.state.busy
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "another question is being answered; try again shortly")
        return StreamingResponse(
            _stream(agent, body, busy, timeout_s),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _run(agent: CiteShieldAgent, body: AskRequest, emit: Callable[..., None]) -> None:
    inputs = {"regulations": body.regulations, "mode": body.mode}
    with traced(emit, body.question, inputs, agent.meta) as e:
        agent.run(body.question, regulations=body.regulations, emit=e, mode=body.mode)


async def _stream(
    agent: CiteShieldAgent, body: AskRequest, busy: threading.Lock, timeout_s: float
) -> AsyncIterator[str]:
    """Stream events until the run ends or ``timeout_s`` passes.

    A timed-out run cannot be interrupted (it runs in a thread); the lock is released only
    when it really finishes, so the next question waits for its memory to be freed.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    def emit(kind: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    async def worker() -> None:
        try:
            await asyncio.to_thread(_run, agent, body, emit)
        except Exception as e:  # report to the client instead of dropping the stream
            log.exception("agent run failed")
            await queue.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            busy.release()
            await queue.put(None)

    task = asyncio.create_task(worker())
    deadline = loop.time() + timeout_s
    try:
        while True:
            remaining = deadline - loop.time()
            try:
                item = await asyncio.wait_for(queue.get(), timeout=max(remaining, 0.001))
            except TimeoutError:
                yield sse("error", {"message": f"no answer within {timeout_s:.0f} s; the run "
                                    "continues in the background and blocks new questions "
                                    "until it ends"})  # fmt: skip
                return
            if item is None:
                break
            yield sse(*item)
    finally:
        if task.done():
            await task


app = create_app()
