"""FastAPI application exposing the LlamaIndex Neo4j agent."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import agent_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting agent service…")
    await agent_service.startup()
    try:
        yield
    finally:
        logger.info("Shutting down agent service…")
        await agent_service.shutdown()


app = FastAPI(title="LlamaIndex Neo4j Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable per-user/per-conversation id")
    message: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sessions")
async def list_sessions() -> dict:
    try:
        return {"sessions": await agent_service.list_sessions()}
    except Exception as exc:  # pragma: no cover
        logger.exception("list sessions failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str) -> dict:
    try:
        messages = await agent_service.get_history(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as exc:  # pragma: no cover
        logger.exception("history fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    try:
        return await agent_service.chat(req.session_id, req.message)
    except Exception as exc:  # pragma: no cover
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    async def gen():
        try:
            async for event in agent_service.stream_chat(req.session_id, req.message):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:  # pragma: no cover
            logger.exception("stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
