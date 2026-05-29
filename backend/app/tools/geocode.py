from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
from timezonefinder import TimezoneFinder


class GeocodePlaceInput(BaseModel):
    place: str = Field(
        description="Human-readable place name, e.g. 'New York City, USA' or 'Rome, Italy'."
    )


def _resolve_place(place: str) -> dict:
    """Internal helper — used by other tools that need geocoding without an LLM hop."""
    try:
        geolocator = Nominatim(user_agent="astroagent-v1/1.0")
        location = geolocator.geocode(place, timeout=10)
        if location is None:
            return {"error": f"Could not find coordinates for '{place}'. Try a more specific name."}
        tf = TimezoneFinder()
        tz = tf.timezone_at(lat=location.latitude, lng=location.longitude)
        if tz is None:
            return {"error": f"Coordinates found for '{place}' but timezone could not be determined."}
        return {
            "lat": location.latitude,
            "lon": location.longitude,
            "display_name": location.address,
            "tz": tz,
        }
    except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as exc:
        return {"error": f"Geocoding service error for '{place}': {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected error geocoding '{place}': {exc}"}


@tool(args_schema=GeocodePlaceInput)
def geocode_place(place: str) -> dict:
    """Resolve a place name to geographic coordinates and IANA timezone string.

    Use this tool when you need the latitude, longitude, and timezone for a
    named location — for example before computing a natal birth chart.

    Returns a dict with keys: lat, lon, display_name, tz.
    On failure returns {"error": "<human-readable reason>"}.
    """
    return _resolve_place(place)
