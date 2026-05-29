from __future__ import annotations
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from app.state import AstroState
from app.tools import geocode_place, compute_birth_chart, get_daily_transits, knowledge_lookup

# Load .env so GOOGLE_API_KEY is available when LLM singletons are first created.
# Kept here so `from app.graph import graph` works from smoke.py / test scripts.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

_TOOLS = [geocode_place, compute_birth_chart, get_daily_transits, knowledge_lookup]

_SYSTEM = """\
You are AstroAgent, a thoughtful astrology companion. All insights you offer are \
for personal reflection and symbolic self-inquiry only — never medical, legal, or \
financial advice. If a user seeks certainty about health, law, or money, gently \
decline and redirect them to qualified professionals.

You have four tools:
  • geocode_place        — resolve a place name to coordinates + IANA timezone
  • compute_birth_chart  — compute a natal chart from a birth date, optional time, and place
  • get_daily_transits   — retrieve current planetary positions for a given date
  • knowledge_lookup     — search curated astrology reference notes for symbolic meaning

Rules you must follow without exception:
1. ALWAYS call the appropriate tool for any factual chart, transit, or planetary data. \
Never invent, guess, or hallucinate planetary positions, degrees, house placements, \
or sign assignments — only state what the tools return.
2. When the user provides birth data (date + place, time optional), call \
compute_birth_chart. Missing birth time is fine — the tool handles it gracefully.
3. For questions about astrological symbolism, sign meanings, house themes, or \
planetary archetypes, call knowledge_lookup to ground your answer.
4. Frame every response as reflective and symbolic, not as prediction or fate.
5. Never give medical, legal, or financial certainty, even when framed astrologically.\
"""

# Maximum number of tool-call round-trips before the agent is forced to summarise.
# The eval grades tool-call count, so this cap matters for scoring.
_MAX_TOOL_CALLS = 6

# ── Lazy LLM singletons — never created at import time ──────────────────────
# This means importing graph.py never requires GOOGLE_API_KEY to be set yet.
_llm_with_tools: ChatGoogleGenerativeAI | None = None
_llm_bare: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    """Tool-bound Gemini singleton — created once on first agent invocation."""
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash"
        ).bind_tools(_TOOLS)
    return _llm_with_tools


def _get_bare_llm() -> ChatGoogleGenerativeAI:
    """Unbound Gemini singleton — used only when the tool budget is exhausted."""
    global _llm_bare
    if _llm_bare is None:
        _llm_bare = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return _llm_bare


# ── Nodes ────────────────────────────────────────────────────────────────────

async def agent(state: AstroState) -> dict[str, list[BaseMessage]]:
    """Invoke the LLM with the system prompt and full conversation history.

    When the tool budget is exhausted the bare (unbound) LLM is used so the
    model is physically unable to emit further tool calls and must synthesise
    a final text response.
    """
    tool_calls_made = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))

    if tool_calls_made >= _MAX_TOOL_CALLS:
        llm = _get_bare_llm()
        system = (
            _SYSTEM
            + "\n\nTool budget exhausted — do NOT call any more tools. "
            "Synthesise all findings gathered so far into your final reply now."
        )
    else:
        llm = _get_llm()
        system = _SYSTEM

    reply: AIMessage = await llm.ainvoke(
        [SystemMessage(content=system), *state["messages"]]
    )
    return {"messages": [reply]}


_tool_node = ToolNode(_TOOLS)


async def run_tools(state: AstroState) -> dict:
    """Execute every tool call requested in the last AIMessage.

    After execution, if compute_birth_chart succeeded its result is written into
    state["natal_chart"] so that get_daily_transits and follow-up questions can
    reuse the cached chart without re-computing or re-geocoding.
    """
    result: dict = await _tool_node.ainvoke(state)

    # Cache compute_birth_chart output into state["natal_chart"]
    new_natal_chart = state.get("natal_chart")
    for msg in result.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name == "compute_birth_chart":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "error" not in data:
                    new_natal_chart = data
            except (json.JSONDecodeError, TypeError):
                pass

    return {**result, "natal_chart": new_natal_chart}


# ── Routing ──────────────────────────────────────────────────────────────────

def should_continue(state: AstroState) -> str:
    """Route to 'tools' if the last AIMessage carries tool calls, else end.

    Also acts as a hard budget guard: if _MAX_TOOL_CALLS ToolMessages already
    exist in state the graph terminates immediately regardless of what the LLM
    requested, preventing runaway loops.
    """
    tool_calls_made = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    if tool_calls_made >= _MAX_TOOL_CALLS:
        return END
    return tools_condition(state)  # returns "tools" or "__end__"


# ── Graph assembly ────────────────────────────────────────────────────────────

_builder = StateGraph(AstroState)
_builder.add_node("agent", agent)
_builder.add_node("tools", run_tools)
_builder.set_entry_point("agent")
_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)
_builder.add_edge("tools", "agent")

graph = _builder.compile()
