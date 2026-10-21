"""FastAPI app. ``POST /ask`` streams the agent's progress as Server-Sent Events:

    event: status       {"stage": "guard" | "research" | "draft", ...}
    event: guard        {"label": "in_scope" | ...}
    event: tool_call    {"name": "retrieve", "input": {...}}
    event: tool_result  {"name": ..., "n_chunks": 8, "sources": [{chunk_id, label, score}]}
    event: note         {"text": "..."}             research notes between tool calls
    event: draft_rejected {"errors": [...]}         validation sent the draft back
    event: final        {"answer": AgentAnswer, "sources": {...}, "usage": {...}, ...}
    event: error        {"message": "..."}

Run: ``uv run uvicorn citeshield.api.main:app``
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from citeshield.agent.graph import CiteShieldAgent

log = logging.getLogger("citeshield.api")

AgentFactory = Callable[[], CiteShieldAgent]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)
    regulations: list[str] | None = Field(None, max_length=5)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def create_app(agent_factory: AgentFactory | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if agent_factory is None:
            from citeshield.agent.factory import build_agent

            factory: AgentFactory = build_agent
        else:
            factory = agent_factory
        # Loading the encoder and reranker takes seconds; do it once, off the event loop.
        app.state.agent = await asyncio.to_thread(factory)
        yield

    app = FastAPI(title="CiteShield", version="0.1.0", lifespan=lifespan)

    def get_agent(request: Request) -> CiteShieldAgent:
        agent: CiteShieldAgent = request.app.state.agent
        return agent

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        agent = get_agent(request)
        return {"status": "ok", **agent.meta}

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
        return StreamingResponse(
            _stream(agent, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _stream(agent: CiteShieldAgent, body: AskRequest) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    def emit(kind: str, payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))

    async def worker() -> None:
        try:
            await asyncio.to_thread(
                agent.run, body.question, regulations=body.regulations, emit=emit
            )
        except Exception as e:  # report to the client instead of dropping the stream
            log.exception("agent run failed")
            await queue.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(worker())
    try:
        while (item := await queue.get()) is not None:
            yield sse(*item)
    finally:
        await task


app = create_app()
