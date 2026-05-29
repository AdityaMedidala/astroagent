from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from app.state import AstroState

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_SYSTEM = (
    "You are AstroAgent, a thoughtful astrology companion. "
    "Your insights are for personal reflection only — never medical, legal, "
    "or financial advice."
)
_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


async def respond(state: AstroState) -> dict[str, list[BaseMessage]]:
    reply: AIMessage = await _llm.ainvoke([SystemMessage(content=_SYSTEM), *state["messages"]])
    return {"messages": [reply]}


_builder = StateGraph(AstroState)
_builder.add_node("respond", respond)
_builder.set_entry_point("respond")
_builder.set_finish_point("respond")

graph = _builder.compile()
