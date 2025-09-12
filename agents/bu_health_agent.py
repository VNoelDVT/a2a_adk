# agents/bu_health_agent.py
import os
import requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Set, Tuple
from google.adk.agents import Agent

try:
    # Let ADK inject this param; it won't appear in the tool schema.
    from google.adk.tools import ToolContext
except Exception:
    ToolContext = None  # type: ignore


# ---------- Agent Card discovery ----------
def _discover(card_env: str, default_card: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a BU service Agent Card from env (BU_CARD_URL) or fallback.
    Returns (base_url, openapi_spec_dict).
    """
    card_url = os.getenv(card_env, default_card)
    card = requests.get(card_url, timeout=5).json()
    svc = card.get("service", {})
    openapi = svc.get("openapi_url") or (
        (svc.get("base_url", "").rstrip("/") + "/openapi.json") if svc.get("base_url") else None
    )
    if not openapi:
        raise RuntimeError(f"[{card_env}] Agent Card invalide.")
    spec = requests.get(openapi, timeout=5).json()
    base = f"{urlparse(openapi).scheme}://{urlparse(openapi).netloc}"
    return base, spec


def _schema_keys(schema: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()

    def walk(s):
        if not isinstance(s, dict):
            return
        for k, v in (s.get("properties") or {}).items():
            keys.add(k)
            walk(v)
        if "items" in s:
            walk(s["items"])
        for alt in ("anyOf", "oneOf", "allOf"):
            for ss in (s.get(alt) or []):
                walk(ss)

    walk(schema or {})
    return keys


def _resolve_by_keys(spec: Dict[str, Any], base: str, required: Set[str]) -> Tuple[str, str]:
    """
    Heuristic: pick the first operation whose request schema contains
    our required keys. Prefer POST and better coverage.
    """
    cands: List[Tuple[int, str, str]] = []
    for path, methods in (spec.get("paths") or {}).items():
        for method, op in (methods or {}).items():
            props: Set[str] = set()
            for _, media in ((op.get("requestBody") or {}).get("content") or {}).items():
                props |= _schema_keys((media or {}).get("schema") or {})
            for p in (op.get("parameters") or []):
                if p.get("in") in ("query", "path") and "name" in p:
                    props.add(p["name"])
                props |= _schema_keys(p.get("schema") or {})
            score = 10 * len(required & props)
            if required and required.issubset(props):
                score += 50
            if method.upper() == "POST":
                score += 2
            if score > 0:
                cands.append((score, method.upper(), f"{base}{path}"))
    if not cands:
        raise RuntimeError("Aucune opération ne correspond aux clés.")
    cands.sort(key=lambda x: (-x[0], x[2]))
    _, m, u = cands[0]
    return m, u


# ---------- BU data fetcher ----------
# Default card: adjust the port/path to your BU service if needed.
_BASE_BU, _SPEC_BU = _discover("BU_CARD_URL", "http://127.0.0.1:9101/.well-known/agent.json")


def _fetch_bu_report(bu_name: str) -> Dict[str, Any]:
    """
    Ask the BU backend for a consolidated report. Falls back to empty sections
    if the service is unavailable, to keep the pipeline running.
    """
    # We’ll look for any operation that accepts at least a 'bu_name' key.
    try:
        method, url = _resolve_by_keys(_SPEC_BU, _BASE_BU, {"bu_name"})
        payload = {"bu_name": bu_name}
        resp = requests.request(method, url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception:
        data = {}

    # Normalize to the structure expected by bu_health_view
    report: Dict[str, Any] = {
        "bu_name": data.get("bu_name") or bu_name,
        "kpis": data.get("kpis") or {
            "revenue_mtd": data.get("revenue_mtd"),
            "revenue_qtd": data.get("revenue_qtd"),
            "gross_margin": data.get("gross_margin"),
            "win_rate": data.get("win_rate"),
        },
        "utilization": data.get("utilization") or {
            "overall": data.get("utilization_overall"),
            "bench": data.get("utilization_bench"),
            "billable": data.get("utilization_billable"),
        },
        "pipeline": data.get("pipeline") or {
            "open_deals": (data.get("open_deals") or data.get("pipeline_open_deals")),
            "value": data.get("pipeline_value"),
            "next_30d": data.get("pipeline_next_30d"),
        },
        "risks": data.get("risks") or [],
        "actions": data.get("actions") or [],
    }
    # Clean empty inner dicts if they are all None
    for section in ("kpis", "utilization", "pipeline"):
        sec = report.get(section) or {}
        if isinstance(sec, dict) and not any(v is not None for v in sec.values()):
            report[section] = {}
    return report


# ---------- Tool exposed to the orchestrator ----------
def bu_overview(bu_name: str, tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Aggregate KPIs for a BU and return a normalized report dict
    expected by bu_health_view (which will render the final text).

    This is an intermediate tool: we KEEP the mini-summary bubble,
    so we do NOT set skip_summarization here.
    """
    report = _fetch_bu_report(bu_name)

    # Mark prerequisite in session state so the final view can enforce order.
    try:
        if tool_context is not None:
            st = getattr(tool_context, "state", None)
            if isinstance(st, dict):
                st["did_bu_overview"] = True
                st.setdefault("flow", "bu")
    except Exception:
        pass

    return report


# Optional sub-agent container
bu_health_agent = Agent(
    name="bu_health_agent",
    model="gemini-1.5-flash-8b",
    instruction="Récupère et agrège les KPIs d'une Business Unit et renvoie un rapport structuré.",
    tools=[bu_overview],
)
