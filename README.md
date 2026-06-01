# ✦ AstroAgent

> **A daily spiritual companion, built with agentic AI.**

AstroAgent is a LangGraph-powered AI astrologer that computes your birth chart, reasons over real planetary data, and answers questions with warmth and care. It’s built to be a conversational guide—grounded in real astronomical math, not hallucination.

---

## 🔮 How It Works

AstroAgent uses a stateful agent graph to route requests, execute tools, and ensure responses are safe, accurate, and perfectly toned.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#F0EDF8', 'primaryTextColor': '#2D2547', 'primaryBorderColor': '#8B7EC8', 'lineColor': '#6B5EA8', 'secondaryColor': '#FDF6E3', 'tertiaryColor': '#FFFFFF'}}}%%
graph TD
    classDef startend fill:#bfb6fc,stroke:#6B5EA8,stroke-width:2px,color:#2C2830,font-weight:bold,rx:10px,ry:10px;
    classDef router fill:#FDF6E3,stroke:#B8892A,stroke-width:2px,color:#2C2830,rx:5px,ry:5px;
    classDef agent fill:#FFFFFF,stroke:#6B5EA8,stroke-width:2px,color:#2C2830,rx:5px,ry:5px;
    classDef tools fill:#F0EDF8,stroke:#8B7EC8,stroke-width:2px,color:#2C2830,rx:5px,ry:5px;
    classDef special fill:#FEF2F2,stroke:#C04A3E,stroke-width:2px,color:#2C2830,rx:5px,ry:5px;

    Start([Start]):::startend --> Router{Intent Router}:::router
    
    Router -- "Off-topic" --> Decline[Decline gracefully]:::special
    Router -- "Astrology" --> SensCheck{Sensitivity Check}:::router
    
    SensCheck -- "Sensitive (HITL)" --> Pause((Human Approval)):::special
    Pause -- "Approved" --> Agent
    Pause -- "Declined" --> End
    
    SensCheck -- "Standard" --> Agent(Reasoning Agent):::agent
    
    Agent -->|Needs data| Tools[(Tools)]:::tools
    Tools -->|Results| Agent
    
    Agent -->|Draft ready| Editor(Tone Editor):::agent
    Decline --> End([End]):::startend
    Editor --> End
```

### ✨ Features
- **Deterministic Math**: Calculates planetary positions accurately offline using `kerykeion` (Swiss Ephemeris).
- **RAG Knowledge Base**: Semantically searches curated astrology notes to stay grounded.
- **Human-in-the-Loop (HITL)**: Automatically detects sensitive topics (health, finance, romance) and pauses for user approval before offering readings.
- **Tone Editor**: A final LLM pass that re-writes the response for a warm, calming, spiritual tone without altering factual astrology.
- **Cross-Session Memory**: SQLite persistence remembers your birth details and previous readings.

---

## 🚀 Quickstart

**Requirements**: Python 3.13, Node.js 18+, a Google Gemini API Key.

### 1. Start the Backend
```bash
cd backend
uv sync
cp .env.example .env  # Add your GOOGLE_API_KEY
uv run uvicorn app.main:app --reload --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) and start exploring!

---

## 📊 Evaluation
AstroAgent is built on a rigorous, eval-driven approach. 
Check out [EVALUATION.md](EVALUATION.md) to see how we track correctness, latency, and cost using a 22-case golden dataset and an LLM-as-a-judge harness.
