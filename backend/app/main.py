from __future__ import annotations
import json
from typing import Any, AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.ai import AIMessageChunk
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.graph import graph
from app.state import AstroState

app = FastAPI(title="AstroAgent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    birth_details: dict[str, Any] | None = None


def _to_messages(raw: list[dict[str, str]]) -> list[BaseMessage]:
    result: list[BaseMessage] = []
    for m in raw:
        if m.get("role") == "assistant":
            result.append(AIMessage(content=m.get("content", "")))
        else:
            result.append(HumanMessage(content=m.get("content", "")))
    return result


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    state: AstroState = {
        "messages": _to_messages(req.messages),
        "birth_details": req.birth_details,
        "natal_chart": None,
        "intent": None,
    }

    async def generate() -> AsyncIterator[dict[str, str]]:
        async for chunk, _ in graph.astream(state, stream_mode="messages"):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield {"data": chunk.content}
        yield {"data": "[DONE]"}

    return EventSourceResponse(generate())
