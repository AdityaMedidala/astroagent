from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from kerykeion import AstrologicalSubjectFactory, KerykeionException

_SIGN_FULL: dict[str, str] = {
    "Ari": "Aries", "Tau": "Taurus", "Gem": "Gemini", "Can": "Cancer",
    "Leo": "Leo", "Vir": "Virgo", "Lib": "Libra", "Sco": "Scorpio",
    "Sag": "Sagittarius", "Cap": "Capricorn", "Aqu": "Aquarius", "Pis": "Pisces",
}

_PLANETS = [
    "sun", "moon", "mercury", "venus", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
]

# Orb thresholds (degrees) for major aspects
_ASPECTS: list[tuple[float, float, str]] = [
    (0.0,   8.0, "conjunct"),
    (60.0,  5.0, "sextile"),
    (90.0,  6.0, "square"),
    (120.0, 6.0, "trine"),
    (180.0, 8.0, "opposite"),
]


def _angular_distance(a: float, b: float) -> float:
    """Shortest arc between two ecliptic positions (0–360°)."""
    diff = abs(a - b) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def _find_natal_house(transit_abs_pos: float, house_cusps: list[dict]) -> str:
    """Return the name of the natal house that contains transit_abs_pos."""
    # Sort cusps by absolute ecliptic position
    cusps = sorted(house_cusps, key=lambda h: h["abs_pos"])
    n = len(cusps)
    for i in range(n):
        start: float = cusps[i]["abs_pos"]
        end: float = cusps[(i + 1) % n]["abs_pos"]
        if end > start:
            if start <= transit_abs_pos < end:
                return cusps[i]["house"]
        else:
            # Arc wraps around 0°/360°
            if transit_abs_pos >= start or transit_abs_pos < end:
                return cusps[i]["house"]
    return cusps[0]["house"]  # fallback


class GetDailyTransitsInput(BaseModel):
    date: str = Field(
        description="Date for transit calculation in YYYY-MM-DD format, e.g. '2026-05-29'."
    )
    natal_chart: Optional[dict] = Field(
        default=None,
        description=(
            "Optional natal chart dict as returned by compute_birth_chart. "
            "When provided, each transiting planet will show which natal house it "
            "currently occupies, and notable aspect contacts with natal planets are listed. "
            "Pass null for a plain transit snapshot with no personalisation."
        ),
    )


@tool(args_schema=GetDailyTransitsInput)
def get_daily_transits(date: str, natal_chart: Optional[dict] = None) -> dict:
    """Retrieve current planetary transit positions for a given date.

    Calculates where each planet sits in the zodiac at noon UTC on the specified
    date using the Swiss Ephemeris (offline mode).

    If a natal_chart dict (from compute_birth_chart) is provided and contains
    house cusp data, each transiting planet also shows which natal house it
    currently occupies. Major aspect contacts (conjunction, sextile, square,
    trine, opposition) between transiting planets and natal planets are listed
    under notable_aspects.

    Returns structured JSON with each planet's current sign, degree, abs_pos,
    and retrograde status, plus optional natal context.
    On failure returns {"error": "<human-readable reason>"}.
    """
    # 1. Validate date
    try:
        d = datetime.date.fromisoformat(date)
    except ValueError:
        return {"error": f"Invalid date '{date}'. Use YYYY-MM-DD format."}

    # 2. Compute planetary positions at noon UTC (Greenwich as reference point)
    try:
        subj = AstrologicalSubjectFactory.from_birth_data(
            name="Transits",
            year=d.year,
            month=d.month,
            day=d.day,
            hour=12,
            minute=0,
            lng=0.0,
            lat=51.4779,   # Greenwich
            tz_str="UTC",
            online=False,
        )
    except KerykeionException as exc:
        return {"error": f"Transit calculation failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected transit error: {exc}"}

    # 3. Determine if natal house mapping is possible
    natal_cusps: list[dict] | None = None
    has_natal_houses = False
    if natal_chart and isinstance(natal_chart, dict):
        if natal_chart.get("time_known") and natal_chart.get("house_cusps"):
            natal_cusps = natal_chart["house_cusps"]
            has_natal_houses = True

    # 4. Build transit entries
    transits_out: dict = {}
    for pname in _PLANETS:
        p = getattr(subj, pname)
        entry: dict = {
            "sign": _SIGN_FULL.get(p.sign, p.sign),
            "sign_short": p.sign,
            "degree": round(float(p.position), 2),
            "abs_pos": round(float(p.abs_pos), 2),
            "retrograde": bool(p.retrograde),
        }
        if has_natal_houses and natal_cusps is not None:
            entry["natal_house"] = _find_natal_house(float(p.abs_pos), natal_cusps)
        transits_out[pname] = entry

    # 5. Find notable aspect contacts with natal planets
    aspect_notes: list[str] = []
    if natal_chart and isinstance(natal_chart, dict) and "planets" in natal_chart:
        natal_planets: dict = natal_chart["planets"]
        for t_name, t_data in transits_out.items():
            for n_name, n_data in natal_planets.items():
                if not isinstance(n_data, dict) or "abs_pos" not in n_data:
                    continue
                arc = _angular_distance(t_data["abs_pos"], float(n_data["abs_pos"]))
                for target, orb, aspect_name in _ASPECTS:
                    if abs(arc - target) <= orb:
                        aspect_notes.append(
                            f"Transit {t_name.capitalize()} {aspect_name} natal "
                            f"{n_name} (orb {abs(arc - target):.1f}°)"
                        )
                        break  # one aspect per pair

    result: dict = {
        "date": date,
        "transits": transits_out,
    }
    if has_natal_houses:
        result["natal_houses_included"] = True
    if aspect_notes:
        result["notable_aspects"] = aspect_notes

    return result
