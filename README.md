# AstroAgent

AstroAgent is an agentic astrology companion built on a LangGraph + FastAPI streaming backend and a React + TypeScript frontend. You give it a birth date and place (time optional), ask about today's sky, or pose a question about sign or house symbolism — and a Gemini-powered agent decides which tools to call, calls them, and streams a reflective interpretation back to you. All astrological insights are framed as symbolic self-inquiry; the system explicitly refuses medical, legal, and financial certainty. The project is eval-driven: a 22-case golden set and a one-command harness with both deterministic checks and an LLM-as-judge layer were built alongside the features. See [EVALUATION.md](EVALUATION.md) for findings.

---

## Architecture

The backend is a single LangGraph state graph. The diagram below is generated from the compiled graph at runtime — not hand-drawn.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	route_intent(route_intent)
	decline_offtopic(decline_offtopic)
	agent(agent)
	tools(tools)
	__end__([<p>__end__</p>]):::last
	__start__ --> route_intent;
	agent -.-> __end__;
	agent -.-> tools;
	route_intent -.-> agent;
	route_intent -.-> decline_offtopic;
	tools --> agent;
	decline_offtopic --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Every request first hits `route_intent`, a cheap structured-output call that classifies the message as one of five intents: `chart_request`, `daily_horoscope`, `freeform`, `offtopic`, or `adversarial`. Off-topic messages short-circuit to `decline_offtopic`, which returns a canned warm redirect without touching any tool or running the main model. Everything else — chart requests, daily horoscopes, freeform symbolism questions, and adversarial probes — enters the `agent ⇄ tools` loop, where the agent decides whether to call tools and the `tools` node executes them and returns results. A hard budget of six tool calls is enforced: once six `ToolMessage`s exist in state, the agent switches to a bare (tool-free) model instance that is physically unable to emit further tool calls and must synthesise a final reply from what it already has.

**The four tools:**

`compute_birth_chart` accepts a birth date, optional time, and place. It geocodes the place via Nominatim + timezonefinder (the `_resolve_place` helper, shared with `geocode_place`), then calls kerykeion's `AstrologicalSubjectFactory.from_birth_data()` in `online=False` mode — no network after geocoding. It returns planetary sign, degree, house (when time is known), and retrograde status for ten planets, plus Ascendant, Midheaven, and twelve house cusps when birth time is available. Results are memoised in a module-level dict keyed on `(normalised_date, normalised_time, normalised_place)`. On a cache hit the tool returns immediately without re-running kerykeion or geocoding (1,366× speedup measured in practice).

`geocode_place` is the standalone wrapper around `_resolve_place`. The agent rarely needs to call it directly — `compute_birth_chart` geocodes internally — but it is available when the user explicitly asks for coordinates or timezone of a place.

`get_daily_transits` computes planetary positions at noon UTC on a given date (defaulting to today) using kerykeion at Greenwich coordinates in offline mode. If a `natal_chart` dict from `compute_birth_chart` is also passed and contains house cusp data, the tool maps each transiting planet to the natal house it currently occupies and lists major aspect contacts (conjunction ≤8°, sextile ≤5°, square ≤6°, trine ≤6°, opposition ≤8°) against natal positions.

`knowledge_lookup` performs semantic search over a curated `notes.md` covering sign elements (fire/earth/air/water), all twelve houses grouped by quadrant, retrograde symbolism, the Sun/Moon/Ascendant triad, and Jupiter and Saturn archetypes. It embeds the query with `gemini-embedding-001`, computes cosine similarity against the pre-embedded note matrix, and returns the top-3 matching chunks.

**The SSE stream** that the frontend consumes carries five typed JSON events: `token` (a text chunk from the agent), `intent` (the router's classification), `tool_start` (tool name + args, emitted when the agent decides to call a tool), `tool_end` (tool name + result, truncated to 1,500 characters if the payload is large), and `done` (stream complete). The React frontend's expandable tool-trace component is driven entirely by `tool_start`/`tool_end` pairs matched by tool name and call order.

---

## Setup

**Requirements:** Python 3.13, Node.js 18+, a Google Gemini API key (free tier works).

### Backend

```bash
cd backend
uv sync                          # installs all dependencies from uv.lock
```

Create `backend/.env`:

```
GOOGLE_API_KEY=your_key_here
```

```bash
uv run uvicorn app.main:app --reload --port 8000
```

The server exposes two endpoints: `GET /health` and `POST /chat` (SSE).

### Frontend

```bash
cd frontend
npm install
npm run dev                      # starts Vite dev server, proxies /chat to localhost:8000
```

Open [http://localhost:5173](http://localhost:5173). Set `VITE_API_URL=http://localhost:8000` in `frontend/.env` if the dev server cannot find the backend.

---

## Running the eval

```bash
cd backend

# Deterministic checks only (intent, tool, budget, well-formed, graceful failure)
uv run python ../eval/runner.py

# Add LLM-as-judge scoring (tone, groundedness, helpfulness, safety)
uv run python ../eval/runner.py --judge

# Judge + print 10 spotcheck verdicts with human-validation template
uv run python ../eval/runner.py --judge --spotcheck

# Limit to N cases (useful for targeted re-checks)
uv run python ../eval/runner.py --limit 4
```

Each run appends a row to `eval/results_log.csv`. For findings, methodology, and the judge validation analysis, see [EVALUATION.md](EVALUATION.md).

---

## Tech choices

**kerykeion over flatlib:** flatlib has not had meaningful maintenance in years and has known incompatibilities with Pydantic v2. kerykeion (v5.12.9) is actively maintained, natively supports Pydantic 2, and ships Swiss Ephemeris binaries via `pyswisseph`, which means all ephemeris calculations run fully offline after geocoding — no third-party astrology API, no network dependency on the calculation path.

**Gemini (gemini-2.0-flash):** specified for the project. `langchain-google-genai` provides clean LangChain tool-binding and structured-output integration. The same model serves as both the intent router (structured output) and the main agent (tool-bound). `gemini-embedding-001` drives the knowledge-lookup semantic search.

**LangGraph:** also specified. Its `StateGraph` with `ToolNode` and `tools_condition` provides the agent loop, budget guard, and streaming infrastructure without bespoke orchestration code.

---

## Known limitations and scope cuts

**Chart cache is process-local.** The memoisation dict lives in module memory. It is not shared across uvicorn workers (each worker builds its own cache independently) and is cleared on every server restart. A production deployment would move this to Redis or a persistent store.

**Single-rater judge validation.** The EV03 spot-check compares judge scores against one human rater across 40 dimension-pairs. 100% within-1 agreement was achieved, but a single rater cannot resolve the systematic +1 offset the judge shows on groundedness and safety. More raters are needed to anchor that calibration difference.

**daily_001 selective-planet omission.** On some runs the agent's daily horoscope response for sign-specific questions names only a subset of the ten returned planets rather than weaving all of them into the interpretation. This is a known gap in the system prompt, not a tool error.

**kerykeion is AGPL-3.0 licensed.** Any distributed derivative of this project must comply with the terms of the GNU Affero General Public License v3.0. Review `kerykeion`'s license before deploying publicly or embedding in a commercial product.
