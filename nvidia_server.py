# server_nvidia.py
import os, json, inspect, re
import unicodedata

from typing import Any, Dict, List, Optional
from math import radians, sin, cos, asin, sqrt

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError

# --- import your existing tools & views ---
from agents.intake_normalizer_agent import detect_and_normalize
from agents.offer_vectorizer_agent import make_offer_vector
from agents.candidate_matcher_agent import match_candidates
from agents.capacity_checker_agent import check_capacity_for_list
from agents.geo_filter_agent import filter_by_geo
from agents.finance_policy_agent import enrich_finance_policy
# from agents.explainability_agent import explain_shortlist  # optional
from agents.bu_health_agent import bu_overview
from orchestrator.ui_contracts import shortlist_view, bu_health_view

# --- NVIDIA/OpenAI client ---
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL    = os.getenv("OPENAI_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
API_KEY  = os.environ.get("OPENAI_API_KEY", "")
client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- knobs ---
GEO_DEFAULT_KM    = int(os.getenv("DEFAULT_MAX_KM", "250"))
MATCH_TOP_N_PRE   = int(os.getenv("MATCH_TOP_N_PRE", "50"))   # pull wide first
FINAL_TOP_N       = int(os.getenv("FINAL_TOP_N", "5"))        # then trim to final N

SYSTEM = (
    "Tu es l’orchestrateur. Suis strictement le pipeline d’outils. "
    "Ne génère pas de texte libre pour l’utilisateur: le rendu final vient du tool `shortlist_view`."
)

# -------------------------- small utils --------------------------
def _params(fn):
    try:
        return [p.name for p in inspect.signature(fn).parameters.values()
                if p.name != "tool_context"]
    except Exception:
        return []

def _maybe_with_tool_context(fn, kwargs: Dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(fn)
        if "tool_context" in sig.parameters and "tool_context" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["tool_context"] = None
    except Exception:
        pass
    return fn(**kwargs)

def _get(d: Dict[str, Any], *keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

def _coerce_list_of_dicts(x: Any) -> List[Dict[str, Any]]:
    """Coerce strings/JSON/wrapped dicts into a list[dict]. Drops junk safely."""
    if x is None:
        return []
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            return _coerce_list_of_dicts(parsed)
        except Exception:
            return []
    if isinstance(x, dict):
        for k in ("results", "items", "shortlist", "candidates", "data"):
            v = x.get(k)
            if isinstance(v, list):
                return _coerce_list_of_dicts(v)
        return []
    if isinstance(x, list):
        out: List[Dict[str, Any]] = []
        for it in x:
            if isinstance(it, dict):
                out.append(it)
            elif isinstance(it, str):
                try:
                    p = json.loads(it)
                    if isinstance(p, dict):
                        out.append(p)
                except Exception:
                    pass
        return out
    return []

# --- header/client sanitization so we don't print a wrong client name ---
def _strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def _contains_ci_noaccents(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return _strip_accents(needle).lower() in _strip_accents(haystack).lower()

def _sanitize_offer_for_view(offer: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """
    Remove client info from the offer if it wasn't explicitly present in the user prompt.
    Prevents headers like '@ Société Générale' when not asked.
    """
    if not isinstance(offer, dict):
        return {}
    client_keys = ("client_id", "client", "customer")
    cid = None
    for k in client_keys:
        if offer.get(k):
            cid = str(offer[k]); break
    if cid and not _contains_ci_noaccents(user_text, cid):
        offer = dict(offer)
        for k in client_keys:
            offer.pop(k, None)
    return offer

# Normalize per-candidate fields so geo can work (promote city to top-level)
def _normalize_shortlist_for_geo(shortlist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normd: List[Dict[str, Any]] = []
    for c in _coerce_list_of_dicts(shortlist):
        r = dict(c)  # shallow copy
        # try top-level first
        city = r.get("city") or r.get("location")
        # then OCR_fields variants
        of = r.get("OCR_fields") or {}
        if not city:
            city = of.get("city") or of.get("base") or of.get("loc")
        # last resort, plausible keys
        if not city:
            for k in ("ville", "town"):
                if of.get(k):
                    city = of[k]; break
        if city:
            r["city"] = city
            r["location"] = city
        normd.append(r)
    return normd

# --------------------- geo signature compatibility ---------------------
def _call_filter_by_geo_signature_aware(shortlist: List[Dict[str, Any]],
                                        city: Optional[str],
                                        max_km: int):
    """
    Call agents.geo_filter_agent.filter_by_geo no matter its signature:
    - def filter_by_geo(payload)
    - def filter_by_geo(payload, tool_context=None)
    - def filter_by_geo(shortlist, location, max_km=400)
    - def filter_by_geo(shortlist, target_city, max_km=400)
    """
    fn = filter_by_geo
    params = _params(fn)
    payload = {"shortlist": shortlist, "location": city, "target_city": city, "max_km": max_km}

    # Single-arg payload style
    if len(params) == 1 and params[0] not in ("shortlist", "location", "target_city", "max_km"):
        name = params[0]
        try:
            return _maybe_with_tool_context(fn, {name: payload})
        except TypeError:
            return fn(payload)

    # Multi-arg explicit fields
    pset = set(params)
    kwargs = {}
    if "shortlist"   in pset: kwargs["shortlist"]   = shortlist
    if "target_city" in pset: kwargs["target_city"] = city
    if "location"    in pset and "target_city" not in pset: kwargs["location"] = city
    if "max_km"      in pset: kwargs["max_km"]      = max_km
    if kwargs:
        return _maybe_with_tool_context(fn, kwargs)

    # Last resort
    try:
        return _maybe_with_tool_context(fn, {"payload": payload})
    except TypeError:
        return fn(payload)

# --------------------- local geo fallback (Haversine) ---------------------
_FR_CITY_COORDS = {
    "lyon":     (45.7640, 4.8357),
    "grenoble": (45.1885, 5.7245),
    "paris":    (48.8566, 2.3522),
    "massy":    (48.7269, 2.2830),
    "brest":    (48.3904, -4.4861),
}
_CITY_ALIASES = {
    # add aliases as needed (accents are stripped before match)
    "lyon": ["lyon"],
    "grenoble": ["grenoble"],
    "paris": ["paris"],
    "massy": ["massy"],
    "brest": ["brest"],
}

def _canonical_city_from_text(user_text: str) -> Optional[str]:
    """Return canonical city key if the prompt explicitly mentions one we know."""
    txt = _strip_accents(user_text or "").lower()
    for canon, aliases in _CITY_ALIASES.items():
        for a in aliases:
            if a in txt:
                return canon  # canonical key matches _FR_CITY_COORDS
    return None

def _canon_to_title(canon: str) -> str:
    return (canon or "").capitalize()

def _norm_city_name(s: str) -> str:
    return _strip_accents((s or "").strip()).lower()

def _haversine_km(p1, p2) -> float:
    (lat1, lon1), (lat2, lon2) = p1, p2
    lat1, lon1, lat2, lon2 = map(radians, [lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371.0088 * c  # km

def _guess_city_from_record(rec: Dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return ""
    for k in ("city", "location"):
        if rec.get(k):
            return str(rec[k])
    of = rec.get("OCR_fields") or {}
    for k in ("city", "base", "loc"):
        if of.get(k):
            return str(of[k])
    return ""

def _geo_filter_local(shortlist: List[Dict[str, Any]], target_city: str, max_km: int) -> List[Dict[str, Any]]:
    """Fallback geo filter: keep only candidates within max_km of target_city."""
    if not shortlist or not target_city:
        return shortlist
    tname = _norm_city_name(target_city)
    if tname not in _FR_CITY_COORDS:
        return shortlist  # unknown city → don't second-guess
    tcoord = _FR_CITY_COORDS[tname]

    out: List[Dict[str, Any]] = []
    for c in shortlist:
        cname = _norm_city_name(_guess_city_from_record(c))
        if not cname:
            continue  # conservative: drop unknown-city candidates
        coord = _FR_CITY_COORDS.get(cname)
        if not coord:
            continue  # conservative: drop unknown coordinates
        if _haversine_km(tcoord, coord) <= max_km:
            out.append(c)
    return out

# --------------------- strict JSON inference for missing fields ---------------------
_CODEFENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

def _json_from_text(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip()
    m = _CODEFENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            return {}
    return {}

def _infer_missing(fields: List[str], context: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """
    Ask the NVIDIA model to fill specific fields (strings) from the user prompt + context.
    Returns dict with those keys; unknowns -> null.
    """
    sys = (
        "Tu extrais des champs manquants à partir d'une requête utilisateur. "
        "Réponds UNIQUEMENT par un JSON, sans texte libre, avec les clés demandées. "
        "Si tu ne sais pas, mets null."
    )
    known = {"offer": context.get("offer") or {}, "vector": context.get("vector") or {}}
    user = (
        "User prompt:\n" + user_text + "\n\n"
        "Champs à renvoyer (tous, même si inconnus): " + ", ".join(fields) + "\n\n"
        "Contexte connu (JSON):\n" + json.dumps(known, ensure_ascii=False)
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.0, top_p=1.0, max_tokens=300,
        )
        content = resp.choices[0].message.content or ""
        data = _json_from_text(content)
        return {k: data.get(k) for k in fields}
    except Exception:
        return {k: None for k in fields}

# -------------------------- main pipeline --------------------------
def orchestrate_once(user_text: str) -> str:
    print("[PIPE] start")
    ctx: Dict[str, Any] = {"offer": None, "vector": None, "shortlist": None}

    # 1) detect
    try:
        out1 = detect_and_normalize(user_text)
        if isinstance(out1, dict):
            ctx["offer"] = out1.get("offer") or {}
        print("[STEP] detect_and_normalize ✓")
    except Exception as e:
        print(f"[STEP] detect_and_normalize ⚠ {e}")
        ctx["offer"] = {}

    # 2) vectorize
    try:
        out2 = make_offer_vector(ctx["offer"])
        if isinstance(out2, dict):
            ctx["vector"] = out2.get("vector") or ctx.get("vector") or {}
            if out2.get("offer"):
                ctx["offer"] = out2["offer"]
        print("[STEP] make_offer_vector ✓")
    except Exception as e:
        print(f"[STEP] make_offer_vector ⚠ {e}")
        ctx["vector"] = ctx.get("vector") or {}

    # 2b) explicit city override from the user prompt (wins over stale/default)
    try:
        explicit_city = _canonical_city_from_text(user_text)
        if explicit_city:
            desired = _canon_to_title(explicit_city)
            # override offer.location
            if _norm_city_name(_get(ctx.get("offer") or {}, "location", "city")) != explicit_city:
                offer = dict(ctx.get("offer") or {})
                offer["location"] = desired
                ctx["offer"] = offer
            # override vector.location
            if _norm_city_name(_get(ctx.get("vector") or {}, "location")) != explicit_city:
                vector = dict(ctx.get("vector") or {})
                vector["location"] = desired
                ctx["vector"] = vector
    except Exception as e:
        print(f"[STEP] explicit city override ⚠ {e}")

    # 3) match (wide first)
    try:
        vector = ctx.get("vector") or {}
        out3 = match_candidates(vector, MATCH_TOP_N_PRE)
        if isinstance(out3, dict) and out3.get("candidates"):
            ctx["shortlist"] = _coerce_list_of_dicts(out3["candidates"])
        elif isinstance(out3, list):
            ctx["shortlist"] = _coerce_list_of_dicts(out3)
        else:
            ctx["shortlist"] = []
        print(f"[STEP] match_candidates ✓ ({len(ctx['shortlist'])} candidats pré-filtre)")
    except Exception as e:
        print(f"[STEP] match_candidates ⚠ {e}")
        ctx["shortlist"] = []

    # 3b) infer missing essentials (start_by, location)
    offer = ctx.get("offer") or {}
    missing: List[str] = []
    if not _get(offer, "start_by", "start", "from"):
        missing.append("start_by")
    if not _get(offer, "location", "city") and not _get(ctx.get("vector") or {}, "location"):
        missing.append("location")
    if missing:
        print(f"[INFER] missing fields -> {missing}")
        inferred = _infer_missing(missing, ctx, user_text)
        if inferred.get("start_by"):
            offer = dict(offer); offer["start_by"] = inferred["start_by"]
        if inferred.get("location"):
            offer = dict(offer); offer["location"] = inferred["location"]
        ctx["offer"] = offer

    # 4) capacity
    try:
        items = _coerce_list_of_dicts(ctx.get("shortlist"))
        start_by = _get(ctx.get("offer") or {}, "start_by", "start", "from") or ""
        out4 = check_capacity_for_list(items, start_by)
        if isinstance(out4, list):
            ctx["shortlist"] = _coerce_list_of_dicts(out4)
        elif isinstance(out4, dict):
            ctx["shortlist"] = _coerce_list_of_dicts(out4.get("shortlist") or out4.get("items"))
        print(f"[STEP] check_capacity_for_list ✓ (len={len(ctx['shortlist'])})")
    except Exception as e:
        print(f"[STEP] check_capacity_for_list ⚠ {e}")

    # 5) geo (normalize each candidate city first!)
    try:
        shortlist = _normalize_shortlist_for_geo(_coerce_list_of_dicts(ctx.get("shortlist")))
        ctx["shortlist"] = shortlist
        city = _get(ctx.get("offer") or {}, "location", "city") or _get(ctx.get("vector") or {}, "location")
        if city and shortlist:
            before = len(shortlist)
            # First call your official tool (signature-agnostic)
            out5 = _call_filter_by_geo_signature_aware(shortlist, city, GEO_DEFAULT_KM)
            if isinstance(out5, dict):
                filtered = _coerce_list_of_dicts(out5.get("shortlist"))
            else:
                filtered = _coerce_list_of_dicts(out5)

            # If no reduction, run a strict local fallback (Haversine)
            if len(filtered) == before:
                filtered2 = _geo_filter_local(shortlist, city, GEO_DEFAULT_KM)
                if filtered2:
                    filtered = filtered2

            ctx["shortlist"] = filtered
            print(f"[STEP] filter_by_geo ✓ (city={city}, len={len(ctx['shortlist'])})")
        else:
            print("[STEP] filter_by_geo ⏭ (no city or empty shortlist)")
    except Exception as e:
        print(f"[STEP] filter_by_geo ⚠ {e}")

    # 6) finance
    try:
        offer = ctx.get("offer") or {}
        shortlist = _coerce_list_of_dicts(ctx.get("shortlist"))
        out6 = enrich_finance_policy(offer, shortlist)
        if isinstance(out6, dict):
            ctx["shortlist"] = _coerce_list_of_dicts(out6.get("shortlist"))
        print(f"[STEP] enrich_finance_policy ✓ (len={len(ctx['shortlist'])})")
    except Exception as e:
        print(f"[STEP] enrich_finance_policy ⚠ {e}")

    # 7) trim to final N after all filters
    if isinstance(ctx.get("shortlist"), list) and FINAL_TOP_N > 0:
        ctx["shortlist"] = ctx["shortlist"][:FINAL_TOP_N]

    # 8) render
    try:
        offer_for_view = _sanitize_offer_for_view(ctx.get("offer") or {}, user_text)
        final = shortlist_view(offer_for_view, _coerce_list_of_dicts(ctx.get("shortlist")))
        print("[VIEW] shortlist_view ✓")
        if isinstance(final, str):
            return final
        if isinstance(final, dict) and final.get("type") == "text":
            return final.get("text", "")
        return final.get("text") if isinstance(final, dict) else str(final)
    except Exception as e:
        print(f"[VIEW] shortlist_view ⚠ {e}")
        return "Aucun résultat exploitable pour l’instant."

# -------------------------- FastAPI app --------------------------
class ChatIn(BaseModel):
    input: str

class ChatOut(BaseModel):
    text: str

app = FastAPI(title="NVIDIA Orchestrator (Deterministic)")

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    try:
        text = orchestrate_once(body.input)
        return ChatOut(text=text)
    except Exception as e:
        return ChatOut(text=f"(Erreur orchestrateur: {type(e).__name__}: {e})")
