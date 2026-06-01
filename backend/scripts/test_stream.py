"""
Start the server, stream /chat, and print every SSE event in order.
Expected sequence: intent → (tool_start → tool_end)* → token… → done
Tokens originate from the editor node (polished reply), not the agent draft.
Usage: cd backend && python scripts/test_stream.py
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PORT = 8989
_BASE = f"http://localhost:{_PORT}"
_MESSAGE = "Compute my chart for 14 March 1879, 11:30 Ulm Germany"
_VENV_PYTHON = str(Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python")


async def _wait_for_server(client: httpx.AsyncClient, retries: int = 30) -> None:
    for _ in range(retries):
        try:
            r = await client.get(f"{_BASE}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError("Server did not start in time.")


async def stream_and_print(client: httpx.AsyncClient) -> None:
    print(f"\nPOST /chat — {_MESSAGE!r}\n")
    print("=" * 68)

    token_chunks: list[str] = []
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

    async with client.stream(
        "POST",
        f"{_BASE}/chat",
        json={"messages": [{"role": "user", "content": _MESSAGE}]},
        timeout=120,
    ) as resp:
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
                    args_str = json.dumps(event.get("args", {}))
                    print(f"[{seq:3d}] {'tool_start':12s}  name={event['name']!r}  args={args_str[:100]}")
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
    print("=" * 68)
    print(f"Total SSE events received: {seq}")


async def main() -> None:
    proc = subprocess.Popen(
        [_VENV_PYTHON, "-m", "uvicorn", "app.main:app",
         "--port", str(_PORT), "--log-level", "error"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    print(f"Server starting on port {_PORT}…", end=" ", flush=True)
    try:
        async with httpx.AsyncClient() as client:
            await _wait_for_server(client)
            print("ready.")
            await stream_and_print(client)
    finally:
        proc.terminate()
        proc.wait()
        print("\nServer stopped.")


if __name__ == "__main__":
    asyncio.run(main())
