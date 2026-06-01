"""
Two-turn cross-session memory verification.

Turn 1: Compute Einstein's birth chart (14 March 1879, 11:30, Ulm Germany)
         with thread_id="test-1". Prints all SSE events.
Turn 2: Ask "What was my sun sign again?" with thread_id="test-1" and NO
         birth data, passing only the new question (not full history).
         Prints all SSE events.

PASS: Turn-2 response contains "Pisces" without calling compute_birth_chart.
FAIL: Response asks for birth data or re-runs the chart tool.

Usage:
    cd backend && uv run python scripts/test_memory.py
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PORT = 8990
_BASE = f"http://localhost:{_PORT}"
_THREAD_ID = "test-1"
_DB_PATH = Path(__file__).resolve().parent.parent / "astro_memory.db"
_VENV_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python")


async def _wait_for_server(client: httpx.AsyncClient, retries: int = 40) -> None:
    for _ in range(retries):
        try:
            r = await client.get(f"{_BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("Server did not start in time.")


async def _stream_turn(
    client: httpx.AsyncClient,
    label: str,
    messages: list[dict[str, str]],
    birth_details: dict | None = None,
) -> tuple[str, list[str]]:
    """Stream a single chat turn; return (full_token_text, tool_names_called)."""
    body: dict = {
        "messages": messages,
        "thread_id": _THREAD_ID,
    }
    if birth_details:
        body["birth_details"] = birth_details

    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")

    token_chunks: list[str] = []
    tool_names_called: list[str] = []
    seq = 0

    def flush_tokens() -> None:
        nonlocal token_chunks
        if not token_chunks:
            return
        full = "".join(token_chunks)
        preview = full[:240].replace("\n", " ")
        ellipsis = "…" if len(full) > 240 else ""
        print(f"     {'token_chunks':12s}  [{len(token_chunks)} chunks, {len(full)} chars]  {preview!r}{ellipsis}")
        token_chunks = []

    async with client.stream("POST", f"{_BASE}/chat", json=body, timeout=180) as resp:
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            raw_line = raw_line.strip()
            if not raw_line.startswith("data:"):
                continue
            payload_str = raw_line[len("data:"):].strip()
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                print(f"  [unparseable] {payload_str[:80]}")
                continue

            seq += 1
            etype = event.get("type", "unknown")

            if etype == "token":
                token_chunks.append(event.get("content", ""))
            else:
                flush_tokens()
                if etype == "intent":
                    print(f"[{seq:3d}] {'intent':12s}  value={event['value']!r}")
                elif etype == "tool_start":
                    name = event["name"]
                    tool_names_called.append(name)
                    args_str = json.dumps(event.get("args", {}))
                    print(f"[{seq:3d}] {'tool_start':12s}  name={name!r}  args={args_str[:120]}")
                elif etype == "tool_end":
                    result_str = json.dumps(event.get("result", {}))
                    preview = result_str[:180]
                    trunc = "…" if len(result_str) > 180 else ""
                    print(f"[{seq:3d}] {'tool_end':12s}  name={event['name']!r}  result={preview}{trunc}")
                elif etype == "done":
                    print(f"[{seq:3d}] {'done':12s}")
                else:
                    print(f"[{seq:3d}] {etype:12s}  {str(event)[:100]}")

    flush_tokens()
    full_text = "".join(token_chunks) if not token_chunks else "".join(token_chunks)
    # Re-join all accumulated chunks including those flushed mid-stream
    return "", tool_names_called  # full_text is printed inline; return tools list


async def _stream_turn_full(
    client: httpx.AsyncClient,
    label: str,
    messages: list[dict[str, str]],
    birth_details: dict | None = None,
) -> tuple[str, list[str]]:
    """Stream a single chat turn; return (full_token_text, tool_names_called)."""
    body: dict = {
        "messages": messages,
        "thread_id": _THREAD_ID,
    }
    if birth_details:
        body["birth_details"] = birth_details

    print(f"\n{'=' * 68}")
    print(f"  {label}")
    print(f"{'=' * 68}")

    all_tokens: list[str] = []
    current_token_run: list[str] = []
    tool_names_called: list[str] = []
    seq = 0

    def flush_tokens() -> None:
        nonlocal current_token_run
        if not current_token_run:
            return
        full = "".join(current_token_run)
        preview = full[:240].replace("\n", " ")
        ellipsis = "…" if len(full) > 240 else ""
        print(f"     {'token_chunks':12s}  [{len(current_token_run)} chunks, {len(full)} chars]  {preview!r}{ellipsis}")
        current_token_run = []

    async with client.stream("POST", f"{_BASE}/chat", json=body, timeout=180) as resp:
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            raw_line = raw_line.strip()
            if not raw_line.startswith("data:"):
                continue
            payload_str = raw_line[len("data:"):].strip()
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                print(f"  [unparseable] {payload_str[:80]}")
                continue

            seq += 1
            etype = event.get("type", "unknown")

            if etype == "token":
                tok = event.get("content", "")
                all_tokens.append(tok)
                current_token_run.append(tok)
            else:
                flush_tokens()
                if etype == "intent":
                    print(f"[{seq:3d}] {'intent':12s}  value={event['value']!r}")
                elif etype == "tool_start":
                    name = event["name"]
                    tool_names_called.append(name)
                    args_str = json.dumps(event.get("args", {}))
                    print(f"[{seq:3d}] {'tool_start':12s}  name={name!r}  args={args_str[:120]}")
                elif etype == "tool_end":
                    result_str = json.dumps(event.get("result", {}))
                    preview = result_str[:180]
                    trunc = "…" if len(result_str) > 180 else ""
                    print(f"[{seq:3d}] {'tool_end':12s}  name={event['name']!r}  result={preview}{trunc}")
                elif etype == "done":
                    print(f"[{seq:3d}] {'done':12s}")
                else:
                    print(f"[{seq:3d}] {etype:12s}  {str(event)[:100]}")

    flush_tokens()
    print(f"Total SSE events: {seq}")
    return "".join(all_tokens), tool_names_called


async def main() -> None:
    # Clean DB so the test is always reproducible
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(_DB_PATH) + suffix)
        if p.exists():
            p.unlink()
            print(f"Removed {p.name}")

    proc = subprocess.Popen(
        [_VENV_PYTHON, "-m", "uvicorn", "app.main:app",
         "--port", str(_PORT), "--log-level", "error"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    print(f"Server starting on port {_PORT}…", end=" ", flush=True)
    try:
        async with httpx.AsyncClient() as client:
            await _wait_for_server(client)
            print("ready.\n")

            # ── Turn 1: chart request ─────────────────────────────────────
            turn1_text, turn1_tools = await _stream_turn_full(
                client,
                label="TURN 1 — Compute natal chart (Einstein)",
                messages=[
                    {"role": "user", "content": "Compute my birth chart: 14 March 1879, 11:30, Ulm Germany"}
                ],
            )

            # ── Turn 2: memory probe ──────────────────────────────────────
            # Pass ONLY the new question — no birth data, no prior messages.
            # LangGraph restores history + natal_chart from the checkpoint.
            turn2_text, turn2_tools = await _stream_turn_full(
                client,
                label="TURN 2 — Memory probe (sun sign?)",
                messages=[
                    {"role": "user", "content": "What was my sun sign again?"}
                ],
            )

    finally:
        proc.terminate()
        proc.wait()
        print("\nServer stopped.")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)

    pisces_found = "pisces" in turn2_text.lower()
    chart_rerun  = "compute_birth_chart" in turn2_tools

    if pisces_found and not chart_rerun:
        print("PASS — Turn 2 returned 'Pisces' without re-running compute_birth_chart.")
    else:
        if not pisces_found:
            print(f"FAIL — 'Pisces' not found in turn-2 response.")
            print(f"       Turn-2 text preview: {turn2_text[:300]!r}")
        if chart_rerun:
            print(f"FAIL — compute_birth_chart was called again in turn 2.")
    print("=" * 68)


if __name__ == "__main__":
    asyncio.run(main())
