# agents/geo_filter_agent.py
from typing import Dict, Any, List

# Minimal distance matrix (km). Add pairs as needed.
_CITY_DISTANCES = {
    ("Paris", "Paris"): 0,
    ("Paris", "Lyon"): 465,
    ("Paris", "Lille"): 225,
    ("Lyon", "Lille"): 680,
    ("Lyon", "Lyon"): 0,
}

def _distance_km(city1: str, city2: str) -> int:
    if not city1 or not city2:
        return 999
    if (city1, city2) in _CITY_DISTANCES:
        return _CITY_DISTANCES[(city1, city2)]
    if (city2, city1) in _CITY_DISTANCES:
        return _CITY_DISTANCES[(city2, city1)]
    return 999

def filter_by_geo(shortlist: List[Dict[str, Any]], target_city: str, radius_km: int = 400, hard: bool = False) -> Dict[str, Any]:
    """
    Annotate each candidate with distance to target_city.
    If hard=False (default), keep everyone and just add flags.
    If hard=True, drop those outside radius_km.
    """
    def _distance_km(city_a: str, city_b: str) -> float:
        # your existing resolver/haversine; return a large number on failure
        try:
            return _resolve_distance(city_a, city_b)  # whatever you currently do
        except Exception:
            return 9e9

    kept = []
    for c in shortlist:
        d = _distance_km(c.get("city", ""), target_city or "")
        info = c | {"geo": {"distance_km": round(d, 1), "within_radius": d <= radius_km}}
        if hard:
            if info["geo"]["within_radius"]:
                kept.append(info)
        else:
            kept.append(info)  # soft mode: keep all, just annotate
    return {"status": "success", "shortlist": kept}
