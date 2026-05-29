from __future__ import annotations
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AstroState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    birth_details: dict | None
    natal_chart: dict | None
    intent: str | None
