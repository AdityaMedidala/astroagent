from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from kerykeion import AstrologicalSubjectFactory, KerykeionException

from app.tools.geocode import _resolve_place

# Kerykeion returns 3-letter sign abbreviations; map to full names for readability.
_SIGN_FULL: dict[str, str] = {
    "Ari": "Aries", "Tau": "Taurus", "Gem": "Gemini", "Can": "Cancer",
    "Leo": "Leo", "Vir": "Virgo", "Lib": "Libra", "Sco": "Scorpio",
    "Sag": "Sagittarius", "Cap": "Capricorn", "Aqu": "Aquarius", "Pis": "Pisces",
}

_PLANETS = [
    "sun", "moon", "mercury", "venus", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
]

_HOUSE_ATTRS = [
    "first_house", "second_house", "third_house", "fourth_house",
    "fifth_house", "sixth_house", "seventh_house", "eighth_house",
    "ninth_house", "tenth_house", "eleventh_house", "twelfth_house",
]


class ComputeBirthChartInput(BaseModel):
    date: str = Field(
        description="Birth date in YYYY-MM-DD format, e.g. '1990-06-15'. Impossible dates such as '1994-02-30' are rejected."
    )
    place: str = Field(
        description="Birth place as a readable city/country string, e.g. 'Rome, Italy' or 'Sydney, Australia'."
    )
    time: Optional[str] = Field(
        default=None,
        description="Birth time in HH:MM 24-hour format, e.g. '14:30'. Omit or pass null if the birth time is unknown.",
    )


@tool(args_schema=ComputeBirthChartInput)
def compute_birth_chart(date: str, place: str, time: Optional[str] = None) -> dict:
    """Compute a natal (birth) astrological chart for a given date, place, and optional time.

    Geocodes the birth place, then calculates planetary positions, houses, and the
    Ascendant using the Swiss Ephemeris (via kerykeion) in fully offline mode.

    If birth time is unknown, planetary sign placements are still returned but
    house positions and the Ascendant are omitted and time_known is set to false.

    Validates the date first — impossible dates such as February 30 or April 31
    are rejected with a clear error message.

    Returns structured JSON with each planet's sign, degree, house, and retrograde
    status, plus Ascendant, Midheaven, and house cusps when time is known.
    On failure returns {"error": "<human-readable reason>"}.
    """
    # 1. Validate and parse date
    try:
        d = datetime.date.fromisoformat(date)
    except ValueError:
        return {
            "error": (
                f"Invalid date '{date}'. Use YYYY-MM-DD and ensure the day actually "
                "exists in that month (e.g. February has at most 28 or 29 days)."
            )
        }

    # 2. Parse optional time
    time_known = False
    hour, minute = 12, 0  # neutral default when time is unknown
    if time:
        try:
            parts = time.strip().split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return {"error": f"Invalid time '{time}'. Hour must be 0–23 and minute 0–59."}
            time_known = True
        except (ValueError, IndexError):
            return {"error": f"Invalid time format '{time}'. Expected HH:MM in 24-hour notation."}

    # 3. Geocode birth place
    geo = _resolve_place(place)
    if "error" in geo:
        return geo

    # 4. Compute natal chart (offline — no network after geocode)
    try:
        subj = AstrologicalSubjectFactory.from_birth_data(
            name="Native",
            year=d.year,
            month=d.month,
            day=d.day,
            hour=hour,
            minute=minute,
            lng=geo["lon"],
            lat=geo["lat"],
            tz_str=geo["tz"],
            online=False,
        )
    except KerykeionException as exc:
        return {"error": f"Chart calculation failed: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected chart error: {exc}"}

    # 5. Build planet entries
    planets_out: dict = {}
    for pname in _PLANETS:
        p = getattr(subj, pname)
        entry: dict = {
            "sign": _SIGN_FULL.get(p.sign, p.sign),
            "sign_short": p.sign,
            "degree": round(float(p.position), 2),
            "abs_pos": round(float(p.abs_pos), 2),
            "retrograde": bool(p.retrograde),
        }
        if time_known:
            entry["house"] = p.house
        planets_out[pname] = entry

    result: dict = {
        "time_known": time_known,
        "birth_place": {
            "display_name": geo["display_name"],
            "lat": geo["lat"],
            "lon": geo["lon"],
            "tz": geo["tz"],
        },
        "planets": planets_out,
    }

    if time_known:
        asc = subj.ascendant
        mc = subj.medium_coeli
        result["ascendant"] = {
            "sign": _SIGN_FULL.get(asc.sign, asc.sign),
            "degree": round(float(asc.position), 2),
            "abs_pos": round(float(asc.abs_pos), 2),
        }
        result["midheaven"] = {
            "sign": _SIGN_FULL.get(mc.sign, mc.sign),
            "degree": round(float(mc.position), 2),
            "abs_pos": round(float(mc.abs_pos), 2),
        }
        # House cusps are needed by get_daily_transits for natal-house transit mapping
        house_cusps: list[dict] = []
        for attr in _HOUSE_ATTRS:
            h = getattr(subj, attr)
            house_cusps.append({
                "house": h.name,
                "sign": _SIGN_FULL.get(h.sign, h.sign),
                "degree": round(float(h.position), 2),
                "abs_pos": round(float(h.abs_pos), 2),
            })
        result["house_cusps"] = house_cusps
    else:
        result["note"] = (
            "Birth time unknown — house positions and Ascendant omitted. "
            "Planetary signs are accurate; Moon degree may be off by up to ~6° "
            "depending on actual birth time."
        )

    return result
