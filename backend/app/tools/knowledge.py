from __future__ import annotations

from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Load .env so GOOGLE_API_KEY is available when the embedder is first used.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

_NOTES_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "notes.md"


def _parse_notes(path: Path) -> list[str]:
    """Split notes.md into sections on ## headings; return each as a standalone chunk."""
    text = path.read_text(encoding="utf-8")
    raw = [s.strip() for s in text.split("##") if s.strip()]
    return [f"## {chunk}" for chunk in raw]


_notes: list[str] = _parse_notes(_NOTES_PATH)

# Lazy globals — initialised on first query to avoid blocking server startup
# and to defer the API call until GOOGLE_API_KEY is confirmed present.
_embedder: GoogleGenerativeAIEmbeddings | None = None
_note_matrix: np.ndarray | None = None  # shape (N, embedding_dim), float32


def _get_embedder() -> GoogleGenerativeAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embedder


def _get_note_matrix() -> np.ndarray:
    """Embed all notes once and cache the result in memory."""
    global _note_matrix
    if _note_matrix is None:
        vecs = _get_embedder().embed_documents(_notes)
        _note_matrix = np.array(vecs, dtype=np.float32)
    return _note_matrix


def _cosine_top_k(query_vec: list[float], matrix: np.ndarray, k: int = 3) -> list[int]:
    """Return indices of the k most cosine-similar rows in matrix to query_vec."""
    q = np.array(query_vec, dtype=np.float32)
    q_normed = q / (np.linalg.norm(q) + 1e-10)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    m_normed = matrix / row_norms
    scores: np.ndarray = m_normed @ q_normed
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices.tolist()


class KnowledgeLookupInput(BaseModel):
    query: str = Field(
        description=(
            "Astrology topic or question to look up, e.g. "
            "'What does Mercury retrograde mean?' or 'Explain the 7th house' "
            "or 'What are water signs?'."
        )
    )


@tool(args_schema=KnowledgeLookupInput)
def knowledge_lookup(query: str) -> dict:
    """Retrieve the top-3 most relevant astrological reference notes for a query.

    Performs semantic search over a curated set of 12 reference notes covering
    the four sign elements (fire/earth/air/water), all 12 houses grouped by
    quadrant, retrograde planet symbolism, the Sun/Moon/Ascendant triad,
    Jupiter and Saturn archetypes, and the reflection-only disclaimer.

    Use this tool to ground answers about astrological symbolism, sign meanings,
    house themes, planetary archetypes, or to retrieve the disclaimer text.

    Returns {"query": ..., "results": [chunk1, chunk2, chunk3]}.
    On failure returns {"error": "<human-readable reason>"}.
    """
    if not query.strip():
        return {"error": "Query must not be empty."}

    try:
        matrix = _get_note_matrix()
        q_vec = _get_embedder().embed_query(query)
        indices = _cosine_top_k(q_vec, matrix, k=3)
        return {
            "query": query,
            "results": [_notes[i] for i in indices],
        }
    except Exception as exc:
        return {"error": f"Knowledge lookup failed: {exc}"}
