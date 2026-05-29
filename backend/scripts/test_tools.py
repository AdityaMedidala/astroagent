"""
Exercise each tool with one valid and one invalid input.
Usage: cd backend && python scripts/test_tools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # backend/ on path

from app.tools.geocode import geocode_place
from app.tools.chart import compute_birth_chart
from app.tools.transits import get_daily_transits
from app.tools.knowledge import knowledge_lookup


def pp(label: str, result: object) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {label}")
    print("=" * 64)
    print(json.dumps(result, indent=2, default=str))


def main() -> None:
    # ── geocode_place ─────────────────────────────────────────────
    pp(
        "geocode_place VALID: 'Rome, Italy'",
        geocode_place.invoke({"place": "Rome, Italy"}),
    )
    pp(
        "geocode_place INVALID: unresolvable name",
        geocode_place.invoke({"place": "xyzzy_invalid_place_99999"}),
    )

    # ── compute_birth_chart ───────────────────────────────────────
    pp(
        "compute_birth_chart VALID: 1990-06-15 14:30 Rome, Italy",
        compute_birth_chart.invoke({"date": "1990-06-15", "time": "14:30", "place": "Rome, Italy"}),
    )
    pp(
        "compute_birth_chart INVALID: impossible date 1994-02-30",
        compute_birth_chart.invoke({"date": "1994-02-30", "time": "12:00", "place": "Chicago, USA"}),
    )

    # ── get_daily_transits ────────────────────────────────────────
    pp(
        "get_daily_transits VALID: 2026-05-29, no natal chart",
        get_daily_transits.invoke({"date": "2026-05-29"}),
    )
    pp(
        "get_daily_transits INVALID: bad date string",
        get_daily_transits.invoke({"date": "not-a-date"}),
    )

    # ── knowledge_lookup ──────────────────────────────────────────
    pp(
        "knowledge_lookup VALID: Mercury retrograde",
        knowledge_lookup.invoke({"query": "What does Mercury retrograde mean?"}),
    )
    pp(
        "knowledge_lookup INVALID: empty query",
        knowledge_lookup.invoke({"query": ""}),
    )


if __name__ == "__main__":
    main()
