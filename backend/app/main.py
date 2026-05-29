from __future__ import annotations
import json
from typing import Any, AsyncIterator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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

# Tool result payloads can be large (12 house cusps, 10 planets, notes…).
# Truncate SSE events to this char limit so the browser stream stays snappy.
_MAX_RESULT_CHARS = 1500


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


def _parse_tool_result(content: str) -> dict:
    """Parse a tool's JSON result; truncate if it exceeds _MAX_RESULT_CHARS."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(content)[:500]}
    result_str = json.dumps(data)
    if len(result_str) <= _MAX_RESULT_CHARS:
        return data
    return {
        "truncated": True,
        "size_chars": len(result_str),
        "preview": result_str[:_MAX_RESULT_CHARS],
    }


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
        async for item in graph.astream(
            state,
            stream_mode=["messages", "updates"],
            config={"recursion_limit": 25},
        ):
            mode, payload = item

            # ── messages mode: individual streamed chunks ─────────────────
            if mode == "messages":
                chunk, metadata = payload
                node: str = metadata.get("langgraph_node", "")

                # Text tokens from the agent — exclude the router's JSON output
                if node == "agent" and isinstance(chunk, AIMessageChunk) and chunk.content:
                    yield {"data": json.dumps({"type": "token", "content": chunk.content})}

                # Tool result arriving from the tools node
                elif node == "tools" and isinstance(chunk, ToolMessage):
                    yield {"data": json.dumps({
                        "type": "tool_end",
                        "name": chunk.name or "",
                        "result": _parse_tool_result(str(chunk.content)),
                    })}

            # ── updates mode: full node output after it finishes ──────────
            elif mode == "updates":

                # Intent label from the router
                if "route_intent" in payload:
                    intent = payload["route_intent"].get("intent")
                    if intent:
                        yield {"data": json.dumps({"type": "intent", "value": intent})}

                # Tool calls decided by the agent
                if "agent" in payload:
                    for msg in payload["agent"].get("messages", []):
                        if isinstance(msg, AIMessage):
                            for tc in (msg.tool_calls or []):
                                name = tc.get("name", "") if isinstance(tc, dict) else tc.name
                                args = tc.get("args", {}) if isinstance(tc, dict) else tc.args
                                yield {"data": json.dumps({
                                    "type": "tool_start",
                                    "name": name,
                                    "args": args,
                                })}

                # Off-topic decline is a static AIMessage (never streams chunks)
                if "decline_offtopic" in payload:
                    for msg in payload["decline_offtopic"].get("messages", []):
                        if isinstance(msg, AIMessage) and msg.content:
                            yield {"data": json.dumps({
                                "type": "token",
                                "content": str(msg.content),
                            })}

        yield {"data": json.dumps({"type": "done"})}

    return EventSourceResponse(generate())
