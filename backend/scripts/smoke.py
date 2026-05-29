"""Run graph directly (no server). Usage: cd backend && python scripts/smoke.py"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/ on path

from langchain_core.messages import BaseMessage, HumanMessage
from app.graph import graph
from app.state import AstroState


async def main() -> None:
    state: AstroState = {
        "messages": [HumanMessage(content="What does it mean to have Sun in Scorpio?")],
        "birth_details": None,
        "natal_chart": None,
        "intent": None,
    }
    print("Calling Gemini via LangGraph graph...")
    result: AstroState = await graph.ainvoke(state)
    last: BaseMessage = result["messages"][-1]
    print("\n--- AstroAgent reply ---")
    print(last.content)
    print("------------------------")


if __name__ == "__main__":
    asyncio.run(main())
