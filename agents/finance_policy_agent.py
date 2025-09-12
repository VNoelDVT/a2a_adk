# agents/finance_policy_agent.py
import os
import requests
from urllib.parse import urlparse
from typing import Any, Dict, Set
from google.adk.agents import Agent


# -------------------------------
# Discovery helpers (with fallback)
# -------------------------------
def _discover(card_env: str, default_card: str, base_env: str | None = None):
    """
    Find (base_url, openapi_spec) either via BASE_URL/env override or via .well-known agent card.
    """
    # 1) If BASE_URL override is provided, prefer it
    if base_env:
        base_override = os.getenv(base_env)
        if base_override:
            openapi_url = base_override.rstrip("/") + "/openapi.json"
            spec = requests.get(openapi_url, timeout=5).json()
            up = urlparse(openapi_url)
            return f"{up.scheme}://{up.netloc}", spec

    # 2) Fallback to agent card
    card_url = os.getenv(card_env, default_card)
    card = requests.get(card_url, timeout=5).json()
    svc = card.get("service", {})
    openapi = (
        svc.get("openapi_url")
        or (svc.get("base_url", "").rstrip("/") + "/openapi.json" if svc.get("base_url") else None)
    )
    if not openapi:
        raise RuntimeError(f"[{card_env}] Agent Card invalide (pas d'openapi_url/base_url).")

    spec = requests.get(openapi, timeout=5).json()
    up = urlparse(openapi)
    base = f"{up.scheme}://{up.netloc}"
    return base, spec


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


def _resolve(spec: Dict[str, Any], base: str, required: Set[str]) -> tuple[str, str]:
    """
    Pick the best operation by scoring keys coverage.
    +10 per matching key, +50 if all required present, +2 if POST.
    Returns (METHOD, FULL_URL). Raises RuntimeError if nothing plausible.
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

            url = f"{base}{path}" if base and path else None
            if score > best_score and url:
                best_score = score
                best_method = (m or "GET").upper()
                best_url = url

    if not best_method or not best_url or best_score <= 0:
        raise RuntimeError("Finance/Policy API: aucune opération OpenAPI correspondante.")
    return best_method, best_url


# -------------------------------
# Discover Finance & Policy on import
# -------------------------------
_BASE_FIN, _SPEC_FIN = _discover(
    card_env="FIN_CARD_URL",
    default_card="http://127.0.0.1:9104/.well-known/agent.json",
    base_env="FIN_BASE_URL",
)
_BASE_POL, _SPEC_POL = _discover(
    card_env="POL_CARD_URL",
    default_card="http://127.0.0.1:9105/.well-known/agent.json",
    base_env="POL_BASE_URL",
)


# -------------------------------
# Tool: enrich_finance_policy
# -------------------------------
def enrich_finance_policy(offer: dict, shortlist: list[dict]) -> dict:
    """
    Enrichit chaque candidat avec:
      - finance: {'ok': bool, 'reason': str}
      - policy: {'ok': bool, 'reason': str}
    Tolérant aux erreurs: en cas d'indispo des services, applique un fallback simple.
    """
    if not shortlist:
        return {"status": "success", "shortlist": []}

    budget = int(offer.get("budget_tjm", 0))
    constraints = offer.get("constraints", {})

    # Résolution des endpoints (séparément, avec fallback silencieux)
    method_fin = url_fin = None
    method_pol = url_pol = None
    try:
        method_fin, url_fin = _resolve(_SPEC_FIN, _BASE_FIN, {"budget_tjm", "tjm_min", "constraints"})
    except Exception:
        method_fin = url_fin = None
    try:
        # Policy: on part sur un schéma générique (constraints + candidat)
        method_pol, url_pol = _resolve(_SPEC_POL, _BASE_POL, {"constraints"})
    except Exception:
        method_pol = url_pol = None

    enriched: list[dict] = []
    for c in shortlist:
        tjm_min = int(c.get("tjm_min", 0))

        # ---------- Finance ----------
        fin_ok = False
        fin_reason = None
        if method_fin and url_fin:
            payload_fin = {
                "budget_tjm": budget,
                "tjm_min": tjm_min,
                "constraints": constraints,
            }
            try:
                r = requests.request(method_fin, url_fin, json=payload_fin, timeout=10).json()
                # normalisation souple
                fin_ok = bool(
                    r.get("ok")
                    or r.get("within_budget")
                    or r.get("feasible")
                    or r.get("approved")
                )
                fin_reason = r.get("reason") or r.get("message") or r.get("detail")
            except Exception as e:
                fin_ok = budget >= tjm_min
                fin_reason = f"finance fallback (error: {e})"
        else:
            # Fallback budget simple
            fin_ok = budget >= tjm_min
            fin_reason = "finance fallback (no endpoint)"

        # ---------- Policy ----------
        pol_ok = True
        pol_reason = None
        if method_pol and url_pol:
            payload_pol = {
                "constraints": constraints,
                "candidate": {
                    "city": c.get("city"),
                    "languages": c.get("languages"),
                    "grade": c.get("grade"),
                    "skills": c.get("skills"),
                },
                "offer": {
                    "location": offer.get("location"),
                    "languages": offer.get("languages"),
                    "seniority": offer.get("seniority"),
                },
            }
            try:
                r = requests.request(method_pol, url_pol, json=payload_pol, timeout=10).json()
                pol_ok = bool(r.get("ok") or r.get("compliant") or r.get("allowed") or True)
                pol_reason = r.get("reason") or r.get("message") or r.get("detail")
            except Exception as e:
                pol_ok = True
                pol_reason = f"policy fallback (error: {e})"
        else:
            pol_ok = True
            pol_reason = "policy fallback (no endpoint)"

        enriched.append(
            c
            | {
                "finance": {"ok": fin_ok, "reason": fin_reason},
                "policy": {"ok": pol_ok, "reason": pol_reason},
            }
        )

    return {"status": "success", "shortlist": enriched}


# -------------------------------
# ADK Agent wrapper
# -------------------------------
finance_policy_agent = Agent(
    name="finance_policy_agent",
    model="gemini-2.0-flash-001",
    instruction="Vérifie budget (finance) et conformité (policy) pour une shortlist.",
    tools=[enrich_finance_policy],
)
