# agents/capacity_checker_agent.py
import os
import requests
from urllib.parse import urlparse
from typing import Any, Dict, Set, Optional, Tuple, List
from google.adk.agents import Agent


# -------------------------------
# .well-known discovery (+ base override)
# -------------------------------
def _discover(card_env: str, default_card: str, base_env: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Try to discover OpenAPI via:
      1) BASE override (e.g. CAP_BASE_URL) -> {base}/openapi.json
      2) Agent Card (service.openapi_url or base_url + /openapi.json)
    Returns (base_url, openapi_spec) or (None, None) on failure.
    """
    # 1) explicit base url (preferred if present)
    if base_env:
        base_override = os.getenv(base_env)
        if base_override:
            try:
                openapi_url = base_override.rstrip("/") + "/openapi.json"
                spec = requests.get(openapi_url, timeout=5).json()
                up = urlparse(openapi_url)
                return f"{up.scheme}://{up.netloc}", spec
            except Exception:
                return None, None

    # 2) agent card
    try:
        card_url = os.getenv(card_env, default_card)
        card = requests.get(card_url, timeout=5).json()
        svc = card.get("service", {})
        openapi = svc.get("openapi_url") or (
            (svc.get("base_url", "").rstrip("/") + "/openapi.json") if svc.get("base_url") else None
        )
        if not openapi:
            return None, None
        spec = requests.get(openapi, timeout=5).json()
        up = urlparse(openapi)
        base = f"{up.scheme}://{up.netloc}"
        return base, spec
    except Exception:
        return None, None


# Lazy globals so import never fails
_BASE: Optional[str] = None
_SPEC: Optional[Dict[str, Any]] = None

def _ensure_discovered() -> bool:
    global _BASE, _SPEC
    if _BASE and _SPEC:
        return True
    base, spec = _discover(
        card_env="CAP_CARD_URL",
        default_card="http://127.0.0.1:9103/.well-known/agent.json",
        base_env="CAP_BASE_URL",
    )
    _BASE, _SPEC = base, spec
    return _BASE is not None and _SPEC is not None


# -------------------------------
# OpenAPI helpers (safe)
# -------------------------------
def _schema_keys(s: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()

    def walk(x: Any):
        if not isinstance(x, dict):
            return
        for k, v in (x.get("properties") or {}).items():
            keys.add(k)
            walk(v)
        if "items" in x:
            walk(x["items"])
        for alt in ("anyOf", "oneOf", "allOf"):
            for ss in (x.get(alt) or []):
                walk(ss)

    walk(s or {})
    return keys


def _props_for_op(op: Dict[str, Any]) -> Set[str]:
    props: Set[str] = set()
    for _, media in ((op.get("requestBody") or {}).get("content") or {}).items():
        props |= _schema_keys((media or {}).get("schema") or {})
    for p in (op.get("parameters") or []):
        if p.get("in") in ("query", "path") and "name" in p:
            props.add(p["name"])
        props |= _schema_keys(p.get("schema") or {})
    return props


def _resolve(spec: Dict[str, Any], base: str, required: Set[str]) -> Optional[Tuple[str, str]]:
    """
    Choose best OpenAPI operation.
    Scoring:
      +10 per required key match
      +50 if all required keys covered
      +2  if method == POST
    Returns (HTTP_METHOD, FULL_URL) or None if nothing plausible.
    """
    best_score = -1
    best_method = None
    best_url = None

    for path, methods in (spec.get("paths") or {}).items():
        for m, op in (methods or {}).items():
            props = _props_for_op(op or {})
            score = 0
            if props:
                score += 10 * len(required & props)
                if required and required.issubset(props):
                    score += 50
            if (m or "").upper() == "POST":
                score += 2
            if score > best_score:
                best_score = score
                best_method = (m or "GET").upper()
                best_url = f"{base}{path}" if base and path else None

    if not best_method or not best_url or best_score <= 0:
        return None
    return best_method, best_url


# -------------------------------
# Tool: check_capacity_for_list
# -------------------------------
def check_capacity_for_list(
    shortlist: List[Dict[str, Any]],
    start_by: str,
    load: int = 80,
    duration_weeks: int = 12,
) -> Dict[str, Any]:
    """
    Enrichit une shortlist avec la capacité (can_staff / reason).
    NOTE: Non-bloquant — en cas d'erreur, on conserve le candidat
    avec can_staff=True + reason explicative, pour éviter les listes vides.
    """
    if not shortlist:
        return {"status": "success", "shortlist": []}

    # Discover lazily; on failure, annotate everyone optimistically.
    if not _ensure_discovered():
        return {
            "status": "success",
            "shortlist": [
                c | {"capacity": {"can_staff": True, "reason": "capacity: discovery failed"}}
                for c in shortlist
            ],
        }

    resolved = _resolve(_SPEC or {}, _BASE or "", {"consultant_id", "start_by", "load", "duration_weeks"})
    if not resolved:
        return {
            "status": "success",
            "shortlist": [
                c | {"capacity": {"can_staff": True, "reason": "capacity: endpoint not found"}}
                for c in shortlist
            ],
        }

    method, url = resolved
    updated: List[Dict[str, Any]] = []

    for c in shortlist:
        cid = c.get("consultant_id")
        if not cid:
            updated.append(c | {"capacity": {"can_staff": True, "reason": "capacity: missing consultant_id"}})
            continue

        payload = {
            "consultant_id": cid,
            "start_by": start_by,
            "load": int(load),
            "duration_weeks": int(duration_weeks),
        }

        try:
            r = requests.request(method, url, json=payload, timeout=10).json()
            updated.append(
                c
                | {
                    "capacity": {
                        # default True so capacity is non-blocking if service returns partial/unknown
                        "can_staff": bool(r.get("can_staff", True)),
                        "reason": r.get("reason", "capacity: ok"),
                    }
                }
            )
        except Exception as e:
            updated.append(c | {"capacity": {"can_staff": True, "reason": f"capacity: error {e}"}})

    return {"status": "success", "shortlist": updated}


# -------------------------------
# ADK Agent wrapper
# -------------------------------
capacity_checker_agent = Agent(
    name="capacity_checker_agent",
    model="gemini-2.0-flash-001",
    instruction="Vérifie la capacité/what-if pour une shortlist de candidats (annotation non bloquante).",
    tools=[check_capacity_for_list],
)
