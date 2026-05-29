"""Run the full agent graph directly (no server).
Usage: cd backend && python scripts/smoke.py
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/ on path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from app.graph import graph
from app.state import AstroState


def _role(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "HUMAN"
    if isinstance(msg, AIMessage):
        return "AI"
    if isinstance(msg, ToolMessage):
        return f"TOOL({msg.name})"
    return type(msg).__name__.upper()


def _body(msg: BaseMessage) -> str:
    """Render message content, abbreviating long tool payloads."""
    if isinstance(msg, AIMessage) and msg.tool_calls:
        calls = ", ".join(
            f"{tc['name']}({list(tc['args'].keys())})" for tc in msg.tool_calls
        )
        text = msg.content or ""
        return f"[tool_calls: {calls}]" + (f"\n{text}" if text else "")
    content = str(msg.content)
    if len(content) > 800:
        content = content[:800] + "…[truncated]"
    return content


async def main() -> None:
    state: AstroState = {
        "messages": [
            HumanMessage(
                content="Compute my birth chart: 14 March 1879, 11:30, Ulm Germany"
            )
        ],
        "birth_details": None,
        "natal_chart": None,
        "intent": None,
    }

    print("Running AstroAgent — birth chart query (forces tool calls)")
    print("=" * 64)

    result: AstroState = await graph.ainvoke(
        state,
        config={"recursion_limit": 25},
    )

    print("\nFULL MESSAGE TRACE")
    print("=" * 64)
    for i, msg in enumerate(result["messages"], 1):
        print(f"\n[{i}] {_role(msg)}")
        print("-" * 48)
        print(_body(msg))

    print("\n" + "=" * 64)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    print(f"Total tool calls made : {len(tool_msgs)}")
    print(f"natal_chart cached    : {'yes' if result.get('natal_chart') else 'no'}")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
