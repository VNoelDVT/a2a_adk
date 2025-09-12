# agents/candidate_matcher_agent.py
import os, requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Set, Tuple
from google.adk.agents import Agent

# ---- Resolver générique (par clés) ----
def _discover(card_env: str, default_card: str) -> Tuple[str, Dict[str, Any]]:
    card_url = os.getenv(card_env, default_card)
    card = requests.get(card_url, timeout=5).json()
    svc = card.get("service", {})
    openapi = svc.get("openapi_url") or (svc.get("base_url","").rstrip("/") + "/openapi.json" if svc.get("base_url") else None)
    if not openapi:
        raise RuntimeError(f"[{card_env}] Agent Card invalide.")
    spec = requests.get(openapi, timeout=5).json()
    base = f"{urlparse(openapi).scheme}://{urlparse(openapi).netloc}"
    return base, spec

def _schema_keys(schema: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    def walk(s):
        if not isinstance(s, dict): return
        for k, v in (s.get("properties") or {}).items():
            keys.add(k); walk(v)
        if "items" in s: walk(s["items"])
        for alt in ("anyOf","oneOf","allOf"):
            for ss in (s.get(alt) or []): walk(ss)
    walk(schema or {}); return keys

def _resolve_by_keys(spec: Dict[str, Any], base: str, required: Set[str]) -> Tuple[str, str]:
    cands = []
    for path, methods in (spec.get("paths") or {}).items():
        for method, op in (methods or {}).items():
            props: Set[str] = set()
            for _, media in ((op.get("requestBody") or {}).get("content") or {}).items():
                props |= _schema_keys((media or {}).get("schema") or {})
            for p in (op.get("parameters") or []):
                if p.get("in") in ("query","path") and "name" in p: props.add(p["name"])
                props |= _schema_keys(p.get("schema") or {})
            score = 10*len(required & props)
            if required and required.issubset(props): score += 50
            if method.upper() == "POST": score += 2
            if score > 0: cands.append((score, method.upper(), f"{base}{path}"))
    if not cands:
        raise RuntimeError("Aucune opération ne correspond aux clés.")
    cands.sort(key=lambda x: (-x[0], x[2]))
    _, m, u = cands[0]
    return m, u

# ---- Wrappers CV search ----
_BASE_CV, _SPEC_CV = _discover("CV_CARD_URL","http://127.0.0.1:9102/.well-known/agent.json")

def _cv_search(must: List[str], nice: List[str], seniority: str, languages: List[str], start_by: str) -> Dict[str, Any]:
    m, u = _resolve_by_keys(_SPEC_CV, _BASE_CV, {"must","nice","seniority","languages","start_by"})
    payload = {
        "must": must or [],
        "nice": nice or [],
        "seniority": seniority,
        "languages": languages or [],
        "start_by": start_by,
    }
    try:
        return requests.request(m, u, json=payload, timeout=10).json()
    except Exception:
        return {"results": []}

# ---- Normalisation & matching ----
def _coerce_offer_struct(x: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepte:
      - {"vector": {...}} (dict englobant)
      - {...} déjà structuré avec must/nice/seniority/languages/start_by (sous-dict)
    """
    if not isinstance(x, dict):
        raise ValueError(f"match_candidates: format vecteur non supporté: {type(x)}")

    if "vector" in x and isinstance(x["vector"], dict):
        return x["vector"]

    keys = {"must","nice","seniority","languages","start_by"}
    if keys & set(x.keys()):
        return x

    raise ValueError("match_candidates: dict reçu mais aucune clé attendue ('vector' ou must/nice/...).")

def match_candidates(vector: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """
    Appel attendu par l'agent: match_candidates(vector)
    'vector' peut être soit le dict englobant {"vector": {...}}, soit le sous-dict structuré.
    """
    v = _coerce_offer_struct(vector)

    must = v.get("must") or []
    nice = v.get("nice") or []
    seniority = v.get("seniority")
    languages = v.get("languages") or []
    start_by = v.get("start_by")
    loc = (v.get("location") or "").lower()

    res = _cv_search(must, nice, seniority, languages, start_by)
    results = res.get("results", []) or []

    enriched: List[Dict[str, Any]] = []
    for r in results:
        city = (r.get("city") or loc) or ""
        distance_km = 5.0 if city.lower() == loc and loc else 800.0
        enriched.append({**r, "distance_km": distance_km})

    enriched.sort(key=lambda x: (-x.get("score", 0), x["distance_km"]))
    top = enriched[: (top_n or 5)]
    return {"status": "success", "candidates": top, "offer_vector": v}

candidate_matcher_agent = Agent(
    name="candidate_matcher_agent",
    model="gemini-1.5-flash-8b",
    instruction="À partir d'un vecteur d'offre, trouve les meilleurs candidats (Top-N) et calcule une distance approximative.",
    tools=[match_candidates],
)
