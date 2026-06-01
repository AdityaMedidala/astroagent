from __future__ import annotations
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.ai import AIMessageChunk
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.graph import compile_graph
from app.state import AstroState

# Tool result payloads can be large (12 house cusps, 10 planets, notes…).
# Truncate SSE events to this char limit so the browser stream stays snappy.
_MAX_RESULT_CHARS = 1500

_DB_PATH = Path(__file__).parent.parent / "astro_memory.db"


@asynccontextmanager
async def lifespan(app_: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(str(_DB_PATH)) as checkpointer:
        app_.state.graph = compile_graph(checkpointer=checkpointer)
        yield


app = FastAPI(title="AstroAgent API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    birth_details: dict[str, Any] | None = None
    thread_id: str | None = None


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
async def chat(req: ChatRequest, request: Request) -> EventSourceResponse:
    g = request.app.state.graph
    thread_id = req.thread_id or str(uuid.uuid4())

    if req.thread_id:
        # Returning (or new) persistent thread.
        # Pass only the new user message — natal_chart and prior messages
        # live in the checkpoint and are restored automatically by LangGraph.
        # natal_chart key is intentionally absent from this dict: LangGraph
        # preserves non-reducer fields from the checkpoint when they are not
        # present in the input update.
        state: AstroState = {
            "messages": _to_messages(req.messages[-1:]),
            "birth_details": req.birth_details,
            "intent": None,
        }
    else:
        # No thread_id — stateless request (ephemeral uuid, no cross-request memory).
        # Full history passed so the agent has context within the single request.
        state = {
            "messages": _to_messages(req.messages),
            "birth_details": req.birth_details,
            "natal_chart": None,
            "intent": None,
        }

    config = {
        "recursion_limit": 25,
        "configurable": {"thread_id": thread_id},
    }

    async def generate() -> AsyncIterator[dict[str, str]]:
        async for item in g.astream(
            state,
            stream_mode=["messages", "updates"],
            config=config,
        ):
            mode, payload = item

            # ── messages mode: individual streamed chunks ─────────────────
            if mode == "messages":
                chunk, metadata = payload
                node: str = metadata.get("langgraph_node", "")

                # Text tokens from the editor — agent draft is internal only
                if node == "editor" and isinstance(chunk, AIMessageChunk) and chunk.content:
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
