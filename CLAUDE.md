# AstroAgent — project memory

## What this is
An agentic astrology companion. Backend: a LangGraph agent graph exposed over FastAPI with streaming. Frontend: a Vite + React + TS chat app. Astrology is for reflection only — never medical/legal/financial certainty. This guardrail must be enforced in the agent and tested in the eval.

## Structure
- backend/  FastAPI + LangGraph agent. venv at backend/.venv
- frontend/ Vite React TS chat UI
- eval/     golden set (JSONL) + one-command harness + scorecard

## Stack decisions (do not change without asking)
- LLM: Google Gemini via langchain-google-genai
- Ephemeris: kerykeion (offline mode: pass lng/lat/tz_str). API uses AstrologicalSubjectFactory.from_birth_data(...). Read the installed package API before writing chart code — do not guess method names.
- Geocoding: geopy (Nominatim) for place->lat/lon, timezonefinder for lat/lon->tz
- Streaming: sse-starlette EventSourceResponse over LangGraph stream_mode

## Conventions
- Type everything (Python type hints, TS strict).
- Small, single-purpose commits with clear messages.
- Every tool handles bad input gracefully and returns structured errors, never raises into the graph.
- When you add a feature, ask whether it needs a golden-set case; if yes, add one.