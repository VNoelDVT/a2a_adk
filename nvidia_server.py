# server_nvidia.py
import os, json, inspect, re
import unicodedata

from typing import Any, Dict, List, Optional, Tuple
from math import radians, sin, cos, asin, sqrt, ceil

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
BASE_URL   = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL      = os.getenv("OPENAI_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", MODEL)  # allow a different model for free chat
API_KEY    = os.environ.get("OPENAI_API_KEY", "")
client     = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- paths/knobs ---
APP_ROOT            = os.path.dirname(os.path.abspath(__file__))
OFFERS_DATA_PATH    = os.getenv("OFFERS_DATA_PATH", "")
BU_DATA_PATH        = os.getenv("BU_DATA_PATH", "")

GEO_DEFAULT_KM      = int(os.getenv("DEFAULT_MAX_KM", "250"))
MATCH_TOP_N_PRE     = int(os.getenv("MATCH_TOP_N_PRE", "50"))
FINAL_TOP_N         = int(os.getenv("FINAL_TOP_N", "5"))

# consultant→missions scoring knobs
CONSULTANT_GEO_KM   = int(os.getenv("CONSULTANT_GEO_KM", str(GEO_DEFAULT_KM)))
MUST_COVERAGE       = float(os.getenv("MUST_COVERAGE", "0.6"))
WEIGHT_GEO          = float(os.getenv("WEIGHT_GEO", "1.2"))
WEIGHT_TJM          = float(os.getenv("WEIGHT_TJM", "0.8"))

SYSTEM = (
    "Tu es l'orchestrateur. Suis strictement le pipeline d'outils. "
    "Ne génère pas de texte libre pour l'utilisateur: le rendu final vient du tool `shortlist_view`."
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

# Normalize per-candidate fields so geo can work
def _normalize_shortlist_for_geo(shortlist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normd: List[Dict[str, Any]] = []
    for c in _coerce_list_of_dicts(shortlist):
        r = dict(c)
        city = r.get("city") or r.get("location")
        of = r.get("OCR_fields") or {}
        if not city:
            city = of.get("city") or of.get("base") or of.get("loc")
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
    fn = filter_by_geo
    params = _params(fn)
    payload = {"shortlist": shortlist, "location": city, "target_city": city, "max_km": max_km}
    if len(params) == 1 and params[0] not in ("shortlist", "location", "target_city", "max_km"):
        name = params[0]
        try:
            return _maybe_with_tool_context(fn, {name: payload})
        except TypeError:
            return fn(payload)
    pset = set(params)
    kwargs = {}
    if "shortlist"   in pset: kwargs["shortlist"]   = shortlist
    if "target_city" in pset: kwargs["target_city"] = city
    if "location"    in pset and "target_city" not in pset: kwargs["location"] = city
    if "max_km"      in pset: kwargs["max_km"]      = max_km
    if kwargs:
        return _maybe_with_tool_context(fn, kwargs)
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
    "lyon": ["lyon"],
    "grenoble": ["grenoble"],
    "paris": ["paris"],
    "massy": ["massy"],
    "brest": ["brest"],
}

def _canonical_city_from_text(user_text: str) -> Optional[str]:
    txt = _strip_accents(user_text or "").lower()
    for canon, aliases in _CITY_ALIASES.items():
        for a in aliases:
            if a in txt:
                return canon
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
    return 6371.0088 * c

def _guess_city_from_record(rec: Dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return ""
    for k in ("city", "location"):
        if rec.get(k):
            return str(rec[k])
    of = rec.get("OCR_fields") or {}
    for k in ("city", "base", "loc"):
        if of.get(k):
            return str(of.get(k))
    return ""

def _offer_city(offer: Dict[str, Any]) -> Optional[str]:
    return offer.get("location") or _get(offer, "city")

def _city_coords(name: Optional[str]):
    if not name: return None
    return _FR_CITY_COORDS.get(_norm_city_name(name))

def _distance_km_between_cities(city_a: Optional[str], city_b: Optional[str]) -> Optional[float]:
    ca, cb = _city_coords(city_a), _city_coords(city_b)
    if not ca or not cb: return None
    return _haversine_km(ca, cb)

def _geo_filter_local(shortlist: List[Dict[str, Any]], target_city: str, max_km: int) -> List[Dict[str, Any]]:
    if not shortlist or not target_city:
        return shortlist
    tname = _norm_city_name(target_city)
    if tname not in _FR_CITY_COORDS:
        return shortlist
    tcoord = _FR_CITY_COORDS[tname]
    out: List[Dict[str, Any]] = []
    for c in shortlist:
        cname = _norm_city_name(_guess_city_from_record(c))
        if not cname: continue
        coord = _FR_CITY_COORDS.get(cname)
        if not coord: continue
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

# --------------------- intent routing helpers ---------------------
def _intent_from_detect_output(out: Dict[str, Any]) -> str:
    """
    Return 'OFFER', 'CONSULTANT', 'BU', or 'OTHER'.
    Only return 'OFFER' when we have clear offer signals; otherwise fall back to 'OTHER'
    so the generic chatbot can answer.
    """
    if not isinstance(out, dict):
        return "OTHER"

    intent = (out.get("intent") or out.get("type") or "").lower()

    # CONSULTANT first
    if out.get("consultant") or out.get("candidate") or intent in {
        "consultant_search", "candidate_to_missions"
    }:
        return "CONSULTANT"

    # BU report
    bu_name = out.get("bu_name") or out.get("bu") or out.get("business_unit")
    if bu_name or intent in {"bu", "business_unit", "bu_report"}:
        return "BU"

    # OFFER only if there are real signals
    offer = out.get("offer") or {}
    has_offer_signals = bool(offer) and any(
        offer.get(k) for k in ("role", "stack", "skills", "location", "city", "start_by", "budget_tjm")
    )
    if has_offer_signals or intent in {"offer", "mission_to_candidates", "offer_to_candidates"}:
        return "OFFER"

    return "OTHER"

def _force_intent_from_text(user_text: str) -> Optional[str]:
    t = _strip_accents(user_text or "").lower()
    if ("consultant" in t or "candidate" in t or "profil" in t or "mon profil" in t) and (
        "mission" in t or "missions" in t or "dispo" in t
    ):
        return "CONSULTANT"
    if re.search(r"\bbu\b|business\s*unit|rapport\s+de\s+sant[eé]", t):
        return "BU"
    return None

# --------------------- offers dataset loader ---------------------
def _load_offers_dataset() -> List[Dict[str, Any]]:
    candidates = []
    if OFFERS_DATA_PATH:
        candidates.append(OFFERS_DATA_PATH)
    candidates += [
        os.path.join(APP_ROOT, "data", "offersJson"),
        os.path.join(APP_ROOT, "data", "offers.json"),
        os.path.join(APP_ROOT, "data", "offers.jsonl"),
        "data/offersJson",
        "data/offers.json",
        "data/offers.jsonl",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if not text:
                        continue
                    if text.startswith("["):
                        return _coerce_list_of_dicts(json.loads(text))
                    # jsonl fallback
                    lines = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                    return _coerce_list_of_dicts(lines)
            except Exception:
                continue
    return []

_SENIORITY_RANK = {"junior": 1, "mid": 2, "intermediate": 2, "senior": 3, "principal": 4, "lead": 4, "staff": 4}

def _norm_token(s: str) -> str:
    return _strip_accents((s or "").strip().lower())

def _normalize_skill_set(items: List[str]) -> set:
    out = set()
    for raw in items or []:
        t = _norm_token(raw)
        if not t: continue
        out.add(t)
        if t == "cloud":
            out.update({"aws", "gcp", "azure"})
    return out

def _seniority_rank(label: Optional[str]) -> int:
    return _SENIORITY_RANK.get(_norm_token(label or ""), 0)

def _skills_overlap(offer: Dict[str, Any], consultant: Dict[str, Any]) -> Tuple[int, int, int, int]:
    must  = _normalize_skill_set(_get(offer, "must") or [])
    stack = _normalize_skill_set(_get(offer, "stack") or [])
    nice  = _normalize_skill_set(_get(offer, "nice") or [])
    cskills = _normalize_skill_set((_get(consultant, "skills") or _get(consultant, "stack") or []))
    return (len(must & cskills), len(must), len(stack & cskills), len(nice & cskills))

def _offer_budget(offer: Dict[str, Any]) -> Optional[float]:
    return offer.get("budget_tjm")

def _consultant_min_rate(consultant: Dict[str, Any]) -> Optional[float]:
    for k in ("desired_tjm", "min_rate", "min_tjm", "tjm_min", "tjm_floor"):
        v = consultant.get(k)
        if isinstance(v, (int, float)): return float(v)
    of = consultant.get("OCR_fields") or {}
    for k in ("min_rate", "min_tjm", "tjm_min", "tjm_floor"):
        v = of.get(k)
        if isinstance(v, (int, float)): return float(v)
    return None

def _geo_distance_score(city_consultant: Optional[str], offer: Dict[str, Any], max_km: int) -> float:
    if offer.get("remote") is True:
        return 1.0
    dist = _distance_km_between_cities(city_consultant, _offer_city(offer))
    if dist is None: return 0.25
    if dist <= 0:    return 1.0
    return max(0.0, 1.0 - (dist / max_km))

def _tjm_score(offer: Dict[str, Any], desired_tjm: Optional[float]) -> float:
    if desired_tjm is None:
        return 0.0
    budget = _offer_budget(offer)
    if budget is None:
        return 0.0
    diff = (budget - desired_tjm) / max(desired_tjm, 1e-6)
    if diff >= 0:
        return min(1.0, diff)
    return max(-1.0, diff)

def _skills_core_score(must_hits: int, stack_hits: int, nice_hits: int) -> float:
    return must_hits * 3.0 + stack_hits * 1.0 + nice_hits * 0.5

def _must_coverage_ok(must_hits: int, must_len: int) -> bool:
    if must_len == 0:
        return True
    threshold = max(1, ceil(MUST_COVERAGE * must_len))
    return must_hits >= threshold

def _score_offer_for_consultant(offer: Dict[str, Any],
                                consultant: Dict[str, Any],
                                consultant_city: Optional[str],
                                desired_tjm: Optional[float]) -> float:
    must_hits, must_len, stack_hits, nice_hits = _skills_overlap(offer, consultant)
    if not _must_coverage_ok(must_hits, must_len):
        return -1e9
    score = _skills_core_score(must_hits, stack_hits, nice_hits)
    if _seniority_rank(consultant.get("grade")) >= _seniority_rank(offer.get("seniority")):
        score += 1.0
    score += WEIGHT_GEO * _geo_distance_score(consultant_city, offer, CONSULTANT_GEO_KM)
    score += WEIGHT_TJM * _tjm_score(offer, desired_tjm)
    return score

def _render_offers_text(consultant: Dict[str, Any], offers: List[Dict[str, Any]]) -> str:
    header_name = consultant.get("name_masked") or consultant.get("consultant_id") or "Consultant"
    header_loc  = consultant.get("location") or _get(consultant, "city")
    header_grade= consultant.get("grade")
    title = f"Top missions pour {header_name}"
    subtitle_parts = [p for p in [header_grade, header_loc] if p]
    if subtitle_parts:
        title += f" ({', '.join(subtitle_parts)})"
    lines = [title, ""]
    for o in offers:
        role = o.get("role") or "Mission"
        loc  = o.get("location") or "-"
        budget = o.get("budget_tjm")
        start  = o.get("start_by") or "-"
        client = o.get("client_id") or "-"
        must = ", ".join(o.get("must") or []) or "-"
        nice = ", ".join(o.get("nice") or []) or "-"
        lines.append(f"• {role} — {loc} (Client {client}, TJM {budget}€)")
        lines.append(f"  Démarrage: {start}")
        lines.append(f"  Must: {must}")
        lines.append(f"  Nice: {nice}")
        lines.append("")
    return "\n".join(lines).strip()

# --------------------- BU dataset loader + canonicalization ---------------------
def _load_bu_dataset() -> Dict[str, Dict[str, Any]]:
    candidates = []
    if BU_DATA_PATH:
        candidates.append(BU_DATA_PATH)
    candidates += [
        os.path.join(APP_ROOT, "data", "business_units.json"),
        os.path.join(APP_ROOT, "data", "business_units"),
        os.path.join(APP_ROOT, "data", "business_units.jsonl"),
        "data/business_units.json",
        "data/business_units",
        "data/business_units.jsonl",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if not text:
                        continue
                    if text.startswith("{"):
                        return json.loads(text)
                    if text.startswith("["):
                        arr = _coerce_list_of_dicts(json.loads(text))
                        out: Dict[str, Dict[str, Any]] = {}
                        for it in arr:
                            name = it.get("name") or it.get("bu_name") or it.get("title")
                            if name: out[str(name)] = it
                        return out
                    out: Dict[str, Dict[str, Any]] = {}
                    for ln in text.splitlines():
                        if not ln.strip(): continue
                        obj = json.loads(ln)
                        name = obj.get("name") or obj.get("bu_name") or obj.get("title")
                        if name: out[str(name)] = obj
                    return out
            except Exception:
                continue
    return {}

def _slugify_bu(s: str) -> str:
    if not isinstance(s, str): return ""
    t = _strip_accents(s).lower()
    t = re.sub(r"\bet\b", "and", t)           # French 'et'
    t = t.replace("&", "and").replace("+", "and")
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t

def _resolve_bu_key(dataset: Dict[str, Dict[str, Any]], query: str) -> Optional[str]:
    if not dataset or not query:
        return None
    slug_map = { _slugify_bu(k): k for k in dataset.keys() }
    q = _slugify_bu(query)
    if q in slug_map:
        return slug_map[q]
    for slug, orig in slug_map.items():
        if q in slug or slug in q:
            return orig
    return None

def _merge_bu_reports(agent_report: Dict[str, Any],
                      file_report: Dict[str, Any],
                      resolved_name: str) -> Dict[str, Any]:
    agent = dict(agent_report or {})
    filed = dict(file_report or {})
    merged = dict(filed)
    merged.update({k: v for k, v in agent.items() if v not in (None, "", [], {})})
    merged["bu_name"] = resolved_name or agent.get("bu_name") or filed.get("bu_name") or resolved_name
    return merged

def _run_bu_flow(user_text: str, detect_out: Dict[str, Any]) -> str:
    bu_name = (detect_out or {}).get("bu_name") or (detect_out or {}).get("bu") or (detect_out or {}).get("business_unit")
    if not bu_name:
        inferred = _infer_missing(["bu_name"], {"offer": {}, "vector": {}}, user_text)
        bu_name = inferred.get("bu_name")
    if not bu_name:
        return "Je n'ai pas identifié la BU visée."

    agent_report: Dict[str, Any] = {}
    try:
        agent_report = bu_overview(bu_name) or {}
    except Exception:
        agent_report = {}

    ds = _load_bu_dataset()
    resolved_key = _resolve_bu_key(ds, bu_name)
    file_report = ds.get(resolved_key, {}) if resolved_key else {}

    def _is_empty(rep: Dict[str, Any]) -> bool:
        if not isinstance(rep, dict) or not rep:
            return True
        needed = ("headcount_by_grade","tjm_median_by_role","salary_avg_by_grade","utilization_pct","bench_pct")
        return not any(k in rep for k in needed)

    if _is_empty(agent_report) and not file_report:
        return f"Rapport de santé — {bu_name}\n\nAucune donnée exploitable."

    report = _merge_bu_reports(agent_report, file_report, resolved_key or bu_name)

    try:
        view = bu_health_view(report)
        if isinstance(view, str): return view
        if isinstance(view, dict) and view.get("type") == "text": return view.get("text", "")
        return view.get("text") if isinstance(view, dict) else str(view)
    except Exception:
        lines = [f"Rapport de santé — {report.get('bu_name','BU')}", ""]
        if file_report:
            hc = file_report.get("headcount_by_grade")
            if hc: lines.append(f"Effectifs: {sum(v for v in hc.values() if isinstance(v, (int,float)))} (par grade: {hc})")
            util = file_report.get("utilization_pct"); bench = file_report.get("bench_pct")
            if util is not None: lines.append(f"Utilisation: {util}%")
            if bench is not None: lines.append(f"Bench: {bench}%")
        return "\n".join(lines) or f"Rapport de santé — {bu_name}"

# --------------------- consultant → missions helpers ---------------------
def _run_consultant_flow(user_text: str, detect_out: Dict[str, Any]) -> str:
    consultant = (
        detect_out.get("consultant")
        or detect_out.get("candidate")
        or {}
    )
    offers_all = _load_offers_dataset()
    if not offers_all:
        return "Aucune mission disponible dans le référentiel."

    c_city = consultant.get("location") or consultant.get("city")
    desired_tjm = _consultant_min_rate(consultant)

    offers = list(offers_all)
    if c_city:
        geo_pref = _geo_filter_local(offers_all, c_city, CONSULTANT_GEO_KM)
        if geo_pref:
            offers = geo_pref

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for off in offers:
        sc = _score_offer_for_consultant(off, consultant, c_city, desired_tjm)
        if sc > -1e8:
            scored.append((sc, off))

    if not scored:
        return "Aucune mission pertinente trouvée pour ce consultant."

    def _start_key(o: Dict[str, Any]) -> str:
        return (o.get("start_by") or "9999-12-31")

    scored.sort(key=lambda t: _start_key(t[1]))
    scored.sort(key=lambda t: t[0], reverse=True)

    top = [o for (_, o) in scored[:FINAL_TOP_N]] if FINAL_TOP_N > 0 else [o for (_, o) in scored]
    return _render_offers_text(consultant, top)

# --------------------- generic chatbot fallback ---------------------
# Strip chain-of-thought / think blocks if a model still emits them
_THINK_INLINE_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_FENCE_RE  = re.compile(r"```(?:xml|html|think)?\s*<\s*think\s*>.*?<\s*/\s*think\s*>\s*```",
                              re.IGNORECASE | re.DOTALL)

def _strip_meta_blocks(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _THINK_INLINE_RE.sub("", s)
    s = _THINK_FENCE_RE.sub("", s)
    return s.strip()

def _chat_fallback(user_text: str, detect_out: Dict[str, Any]) -> str:
    """Use a general-purpose answer when the intent isn't one of the two flows."""
    system = (
        "Tu es un assistant utile et concis. Réponds dans la langue du message. "
        "N'affiche JAMAIS de balises <think> ni d'analyse interne. "
        "Pas de code fence inutile."
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            top_p=1.0,
            max_tokens=800,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _strip_meta_blocks(content)
    except Exception as e:
        return f"(Assistant: impossible de répondre — {type(e).__name__}: {e})"

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
        out1 = {}
        ctx["offer"] = {}

    # ---- route by intent (two special flows; else → chatbot) ----
    intent = _intent_from_detect_output(out1 or {})
    forced = _force_intent_from_text(user_text)  # can force CONSULTANT/BU
    if forced:
        intent = forced

    if intent == "CONSULTANT":
        print("[ROUTE] consultant → missions")
        return _run_consultant_flow(user_text, out1 or {})

    elif intent == "OFFER":
        print("[ROUTE] mission → candidats")
        # continue into the OFFER pipeline below

    else:
        # 'OTHER' or 'BU' → chatbot fallback (as requested)
        print("[ROUTE] generic assistant fallback")
        return _chat_fallback(user_text, out1 or {})

    # -------------------- OFFER → CANDIDATES PIPELINE --------------------
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

    # explicit city override from the prompt (e.g., "Massy", "Lyon", …)
    try:
        explicit_city = _canonical_city_from_text(user_text)
        if explicit_city:
            desired = _canon_to_title(explicit_city)
            if _norm_city_name(_get(ctx.get("offer") or {}, "location", "city")) != explicit_city:
                offer = dict(ctx.get("offer") or {}); offer["location"] = desired; ctx["offer"] = offer
            if _norm_city_name(_get(ctx.get("vector") or {}, "location")) != explicit_city:
                vector = dict(ctx.get("vector") or {}); vector["location"] = desired; ctx["vector"] = vector
    except Exception as e:
        print(f"[STEP] explicit city override ⚠ {e}")

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

    try:
        shortlist = _normalize_shortlist_for_geo(_coerce_list_of_dicts(ctx.get("shortlist")))
        ctx["shortlist"] = shortlist
        city = _get(ctx.get("offer") or {}, "location", "city") or _get(ctx.get("vector") or {}, "location")
        if city and shortlist:
            before = len(shortlist)
            out5 = _call_filter_by_geo_signature_aware(shortlist, city, GEO_DEFAULT_KM)
            if isinstance(out5, dict):
                filtered = _coerce_list_of_dicts(out5.get("shortlist"))
            else:
                filtered = _coerce_list_of_dicts(out5)
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

    try:
        offer = ctx.get("offer") or {}
        shortlist = _coerce_list_of_dicts(ctx.get("shortlist"))
        out6 = enrich_finance_policy(offer, shortlist)
        if isinstance(out6, dict):
            ctx["shortlist"] = _coerce_list_of_dicts(out6.get("shortlist"))
        print(f"[STEP] enrich_finance_policy ✓ (len={len(ctx['shortlist'])})")
    except Exception as e:
        print(f"[STEP] enrich_finance_policy ⚠ {e}")

    if isinstance(ctx.get("shortlist"), list) and FINAL_TOP_N > 0:
        ctx["shortlist"] = ctx["shortlist"][:FINAL_TOP_N]

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
        return "Aucun résultat exploitable pour l'instant."

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

@app.get("/bu-data")
def get_bu_data():
    """Return current BU data from the JSON file"""
    print("[BU-DATA] GET /bu-data endpoint called")
    
    try:
        # Debug: Show what paths we're checking
        candidates = []
        if BU_DATA_PATH:
            candidates.append(BU_DATA_PATH)
            print(f"[BU-DATA] Checking BU_DATA_PATH: {BU_DATA_PATH}")
        
        default_paths = [
            os.path.join(APP_ROOT, "data", "business_units.json"),
            os.path.join(APP_ROOT, "data", "business_units"),
            os.path.join(APP_ROOT, "data", "business_units.jsonl"),
            "data/business_units.json",
            "data/business_units",
            "data/business_units.jsonl",
        ]
        candidates.extend(default_paths)
        
        print(f"[BU-DATA] APP_ROOT is: {APP_ROOT}")
        print(f"[BU-DATA] Current working directory: {os.getcwd()}")
        print(f"[BU-DATA] Will check these paths in order:")
        for i, path in enumerate(candidates, 1):
            exists = os.path.exists(path)
            print(f"[BU-DATA]   {i}. {path} - {'EXISTS' if exists else 'NOT FOUND'}")
        
        dataset = _load_bu_dataset()
        if not dataset:
            print("[BU-DATA] _load_bu_dataset returned empty - using fallback data")
            # Return your hardcoded fallback data here
            return {
                "Creative Tech": {
                    "headcount_by_grade": { "Junior": 35, "Senior": 55, "Principal": 12, "Manager": 8 },
                    # ... rest of your data
                }
            }
        
        print(f"[BU-DATA] Successfully loaded {len(dataset)} business units from file")
        return dataset
        
    except Exception as e:
        error_msg = f"Could not load BU data: {type(e).__name__}: {e}"
        print(f"[BU-DATA] ERROR: {error_msg}")
        return {"error": error_msg}

# -------------------------- Enhanced UI with Gemini-inspired Design --------------------------
_UI_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Devoteam HR Dashboard</title>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
  <style>
    :root {
      /* Gemini-inspired color palette */
      --primary: #1a73e8;
      --primary-dark: #1557b0;
      --primary-light: #4285f4;
      --secondary: #34a853;
      --accent: #ea4335;
      --warning: #fbbc04;
      --purple: #9334e4;
      --teal: #0d9488;
      
      /* Backgrounds & surfaces */
      --bg-primary: #0f1419;
      --bg-secondary: #1a1f2e;
      --bg-tertiary: #252b3a;
      --surface: rgba(255, 255, 255, 0.05);
      --surface-hover: rgba(255, 255, 255, 0.08);
      
      /* Text colors */
      --text-primary: #ffffff;
      --text-secondary: rgba(255, 255, 255, 0.7);
      --text-muted: rgba(255, 255, 255, 0.5);
      
      /* Borders & shadows */
      --border: rgba(255, 255, 255, 0.1);
      --border-hover: rgba(255, 255, 255, 0.2);
      --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 20px 60px rgba(0, 0, 0, 0.4);
      
      --radius: 12px;
      --radius-lg: 16px;
      --sidebar-width: 320px;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; }
    
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
      color: var(--text-primary);
      overflow-x: hidden;
    }

    /* Sidebar */
    .sidebar {
      position: fixed;
      top: 0;
      left: 0;
      width: var(--sidebar-width);
      height: 100vh;
      background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
      border-right: 1px solid var(--border);
      transform: translateX(-100%);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 1000;
      overflow-y: auto;
    }

    .sidebar.open {
      transform: translateX(0);
    }

    .sidebar-header {
      padding: 24px;
      border-bottom: 1px solid var(--border);
    }

    .sidebar-title {
      font-size: 18px;
      font-weight: 600;
      margin: 0 0 8px;
      background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .sidebar-subtitle {
      font-size: 12px;
      color: var(--text-muted);
      margin: 0;
    }

    .conversation-history {
      padding: 16px;
    }

    .history-item {
      padding: 12px;
      margin-bottom: 8px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      cursor: pointer;
      transition: all 0.2s ease;
    }

    .history-item:hover {
      background: var(--surface-hover);
      border-color: var(--border-hover);
      transform: translateY(-1px);
    }

    .history-item-title {
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 4px;
      color: var(--text-primary);
    }

    .history-item-preview {
      font-size: 11px;
      color: var(--text-muted);
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .history-item-time {
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 8px;
    }

    /* Sidebar footer */
    .sidebar-footer {
      position: sticky;
      bottom: 0;
      background: linear-gradient(180deg, transparent 0%, var(--bg-tertiary) 30%);
      padding: 20px 16px;
      border-top: 1px solid var(--border);
      margin-top: auto;
    }

    .powered-by {
      text-align: center;
    }

    .powered-by-text {
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .logos-container {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .logo-item {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      transition: all 0.2s ease;
    }

    .logo-item:hover {
      background: var(--surface-hover);
      border-color: var(--border-hover);
      transform: translateY(-1px);
    }

    .nvidia-logo {
      width: 32px;
      height: 10px;
    }

    .llama-logo {
      width: 16px;
      height: 16px;
    }

    .logo-text {
      font-size: 10px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .logo-separator {
      font-size: 10px;
      color: var(--text-muted);
      opacity: 0.5;
    }

    /* Main content */
    .main {
      margin-left: 0;
      transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      min-height: 100vh;
    }

    .main.sidebar-open {
      margin-left: var(--sidebar-width);
    }

    .header {
      padding: 20px 24px;
      background: rgba(255, 255, 255, 0.02);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(20px);
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .header-content {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .menu-button {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px;
      cursor: pointer;
      transition: all 0.2s ease;
      color: var(--text-primary);
    }

    .menu-button:hover {
      background: var(--surface-hover);
      border-color: var(--border-hover);
    }

    .app-title {
      font-size: 24px;
      font-weight: 700;
      background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }

    /* Action cards */
    .action-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 20px;
      margin-bottom: 32px;
    }

    .action-card {
      background: linear-gradient(135deg, var(--surface) 0%, rgba(255, 255, 255, 0.02) 100%);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }

    .action-card::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
      opacity: 0;
      transition: opacity 0.3s ease;
    }

    .action-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-lg);
      border-color: var(--border-hover);
    }

    .action-card:hover::before {
      opacity: 1;
    }

    .card-header {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }

    .card-icon {
      width: 24px;
      height: 24px;
      opacity: 0.8;
    }

    .card-title {
      font-size: 16px;
      font-weight: 600;
      margin: 0;
      color: var(--text-primary);
    }

    .card-badge {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      font-size: 10px;
      font-weight: 500;
      padding: 4px 8px;
      border-radius: 12px;
      margin-left: auto;
    }

    textarea {
      width: 100%;
      height: 100px;
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 16px;
      color: var(--text-primary);
      font-family: inherit;
      font-size: 14px;
      resize: vertical;
      transition: all 0.2s ease;
    }

    textarea:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.1);
    }

    textarea::placeholder {
      color: var(--text-muted);
    }

    .button-row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 16px;
    }

    .btn {
      background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
      color: white;
      border: none;
      border-radius: var(--radius);
      padding: 10px 20px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }

    .btn::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
      transition: left 0.5s;
    }

    .btn:hover::before {
      left: 100%;
    }

    .btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 20px rgba(26, 115, 232, 0.3);
    }

    .btn-secondary {
      background: var(--surface);
      color: var(--text-primary);
      border: 1px solid var(--border);
    }

    .btn-secondary:hover {
      background: var(--surface-hover);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .status {
      font-size: 12px;
      color: var(--text-muted);
      margin-left: auto;
    }

    /* BU Health Section */
    .bu-health-section {
      background: linear-gradient(135deg, var(--surface) 0%, rgba(255, 255, 255, 0.02) 100%);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
      margin-bottom: 32px;
      position: relative;
      overflow: hidden;
    }

    .bu-health-section::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--secondary) 0%, var(--teal) 50%, var(--purple) 100%);
    }

    .section-title {
      font-size: 20px;
      font-weight: 700;
      margin: 0 0 20px;
      background: linear-gradient(135deg, var(--secondary) 0%, var(--teal) 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .bu-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }

    .bu-card {
      background: var(--bg-secondary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      transition: all 0.3s ease;
      cursor: pointer;
      position: relative;
    }

    .bu-card:hover {
      transform: translateY(-2px);
      box-shadow: var(--shadow);
      border-color: var(--border-hover);
    }

    .bu-card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }

    .bu-name {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .health-indicator {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    .health-good { background: var(--secondary); }
    .health-warning { background: var(--warning); }
    .health-critical { background: var(--accent); }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .pyramid-container {
      height: 150px;
      margin: 16px 0;
    }

    .metrics-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 12px;
    }

    .metric {
      text-align: center;
    }

    .metric-value {
      font-size: 18px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .metric-label {
      font-size: 11px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    /* Response area */
    .response-section {
      background: linear-gradient(135deg, var(--surface) 0%, rgba(255, 255, 255, 0.02) 100%);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 24px;
    }

    .response-content {
      background: var(--bg-primary);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      min-height: 120px;
      white-space: pre-wrap;
      font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
      font-size: 13px;
      line-height: 1.6;
      color: var(--text-secondary);
    }

    .response-content.has-content {
      color: var(--text-primary);
    }

    /* Loading animation */
    .loading {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--primary);
    }

    .loading::after {
      content: '';
      width: 16px;
      height: 16px;
      border: 2px solid transparent;
      border-top: 2px solid currentColor;
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Responsive design */
    @media (max-width: 768px) {
      .container {
        padding: 16px;
      }
      
      .action-grid {
        grid-template-columns: 1fr;
      }
      
      .bu-grid {
        grid-template-columns: 1fr;
      }
      
      .sidebar {
        width: 280px;
      }
      
      :root {
        --sidebar-width: 280px;
      }
    }
  </style>
</head>
<body>
  <!-- Sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <h2 class="sidebar-title">Historique</h2>
      <p class="sidebar-subtitle">Conversations récentes</p>
    </div>
    <div class="conversation-history" id="conversationHistory">
      <!-- History items will be populated here -->
    </div>
  </div>

  <!-- Main content -->
  <div class="main" id="main">
    <!-- Header -->
    <div class="header">
      <div class="header-content">
        <button class="menu-button" onclick="toggleSidebar()">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
          </svg>
        </button>
        <h1 class="app-title">Devoteam HR Dashboard</h1>
      </div>
    </div>

    <div class="container">
      <!-- Action Cards -->
      <div class="action-grid">
        <div class="action-card">
          <div class="card-header">
            <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>
              <circle cx="9" cy="7" r="4"></circle>
              <path d="M22 21v-2a4 4 0 0 0-3-3.87"></path>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            <h3 class="card-title">Mission → Candidats</h3>
            <span class="card-badge">Use case #1</span>
          </div>
          <textarea id="ta-offer" placeholder="Ex: J'ai une mission Data Engineer à Lyon qui commence en octobre, qui est dispo ?"></textarea>
          <div class="button-row">
            <button class="btn" onclick="send('ta-offer')">Envoyer</button>
            <span id="st-offer" class="status"></span>
          </div>
        </div>

        <div class="action-card">
          <div class="card-header">
            <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <h3 class="card-title">Consultant → Missions</h3>
            <span class="card-badge">Use case #2</span>
          </div>
          <textarea id="ta-consultant" placeholder="Ex: J'ai un consultant senior Python/Cloud basé à Paris, quelles missions sont dispo ?"></textarea>
          <div class="button-row">
            <button class="btn" onclick="send('ta-consultant')">Envoyer</button>
            <span id="st-consultant" class="status"></span>
          </div>
        </div>

        <div class="action-card">
          <div class="card-header">
            <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
            <h3 class="card-title">Chat libre</h3>
            <span class="card-badge">Assistant</span>
          </div>
          <textarea id="ta-free" placeholder="Posez n'importe quelle question… L'assistant détecte automatiquement le meilleur traitement."></textarea>
          <div class="button-row">
            <button class="btn" onclick="send('ta-free')">Envoyer</button>
            <span id="st-free" class="status"></span>
          </div>
        </div>
      </div>

      <!-- BU Health Section -->
      <div class="bu-health-section">
        <h2 class="section-title">BU Health Dashboard</h2>
        <div class="button-row">
          <button class="btn btn-secondary" onclick="openBUHealth()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;">
              <path d="M3 3v18h18"></path>
              <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"></path>
            </svg>
            Analyser la santé des BU
          </button>
          <button class="btn btn-secondary" onclick="loadBusinessUnitsData()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 8px;">
              <polyline points="23 4 23 10 17 10"></polyline>
              <polyline points="1 20 1 14 7 14"></polyline>
              <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"></path>
            </svg>
            Actualiser les données
          </button>
        </div>
        <div class="bu-grid" id="buGrid">
          <!-- BU cards will be populated here -->
        </div>
      </div>

      <!-- Response Section -->
      <div class="response-section">
        <h2 class="section-title">Réponse</h2>
        <div id="out" class="response-content">La réponse apparaîtra ici…</div>
      </div>
    </div>
  </div>

  <script>
    // Business Units data - will be loaded dynamically from the server
    let businessUnitsData = {};

    let conversationHistory = JSON.parse(localStorage.getItem('conversationHistory') || '[]');
    let sidebarOpen = false;

    // Load business units data from the server
    window.loadBusinessUnitsData = async function() {
      try {
        console.log('[BU-UI] Loading business units data from server...');
        const response = await fetch('/bu-data');
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
          console.error('[BU-UI] Server returned error:', data.error);
          return;
        }
        
        businessUnitsData = data;
        console.log(`[BU-UI] Successfully loaded ${Object.keys(businessUnitsData).length} business units`);
        
        // Clear existing BU cards and create new ones
        const container = document.getElementById('buGrid');
        container.innerHTML = '';
        createBUCards();
        
      } catch (error) {
        console.error('[BU-UI] Failed to load BU data from server:', error);
      }
    }

    // Initialize UI
    document.addEventListener('DOMContentLoaded', function() {
      console.log('[BU-UI] DOM loaded, initializing...');
      updateConversationHistory();
      loadBusinessUnitsData(); // Load BU data first, then create cards
    });

    // Sidebar functions - make global for onclick handlers
    window.toggleSidebar = function() {
      sidebarOpen = !sidebarOpen;
      const sidebar = document.getElementById('sidebar');
      const main = document.getElementById('main');
      
      if (sidebarOpen) {
        sidebar.classList.add('open');
        main.classList.add('sidebar-open');
      } else {
        sidebar.classList.remove('open');
        main.classList.remove('sidebar-open');
      }
    }

    // Conversation history functions
    function addToHistory(input, output) {
      const historyItem = {
        id: Date.now(),
        input: input.substring(0, 100),
        output: output.substring(0, 200),
        timestamp: new Date().toISOString(),
        fullInput: input,
        fullOutput: output
      };
      
      conversationHistory.unshift(historyItem);
      if (conversationHistory.length > 20) {
        conversationHistory = conversationHistory.slice(0, 20);
      }
      
      localStorage.setItem('conversationHistory', JSON.stringify(conversationHistory));
      updateConversationHistory();
    }

    function updateConversationHistory() {
      const container = document.getElementById('conversationHistory');
      
      if (conversationHistory.length === 0) {
        container.innerHTML = `
          <div style="text-align: center; padding: 20px; color: var(--text-muted);">
            <p>Aucune conversation pour l'instant</p>
            <p style="font-size: 12px;">Vos échanges apparaîtront ici</p>
          </div>
        `;
        return;
      }

      container.innerHTML = conversationHistory.map(item => `
        <div class="history-item" onclick="loadFromHistory('${item.id}')">
          <div class="history-item-title">${item.input}</div>
          <div class="history-item-preview">${item.output}</div>
          <div class="history-item-time">${new Date(item.timestamp).toLocaleString('fr-FR')}</div>
        </div>
      `).join('');
    }

    window.loadFromHistory = function(id) {
      const item = conversationHistory.find(h => h.id == id);
      if (item) {
        document.getElementById('out').textContent = item.fullOutput;
        document.getElementById('out').classList.add('has-content');
      }
    }

    // BU Health functions
    function createBUCards() {
  const container = document.getElementById('buGrid');
  
  Object.entries(businessUnitsData).forEach(([buName, data]) => {
    const health = calculateBUHealth(data);
    const card = createBUCard(buName, data, health);
    container.appendChild(card);
  });

  // Draw pyramid charts after cards are created
  setTimeout(() => {
    Object.entries(businessUnitsData).forEach(([buName, data]) => {
      const canvasId = `pyramid-${buName.replace(/\s+/g, '')}`;
      drawPyramidChart(canvasId, data.headcount_by_grade);
    });
  }, 100);
}

    function calculateBUHealth(data) {
      const headcount = data.headcount_by_grade;
      const total = Object.values(headcount).reduce((sum, count) => sum + count, 0);
      const juniorRatio = headcount.Junior / total;
      const seniorRatio = (headcount.Senior + headcount.Principal + headcount.Manager) / total;
      
      let warnings = [];
      let healthStatus = 'good';
      
      if (juniorRatio < 0.3) {
        warnings.push("⚠️ Faible ratio de juniors");
        healthStatus = 'warning';
      }
      
      if (seniorRatio > 0.7) {
        warnings.push("⚠️ Structure top-heavy");
        healthStatus = 'warning';
      }
      
      if (data.utilization_pct < 75) {
        warnings.push("⚠️ Utilisation faible");
        healthStatus = 'warning';
      }
      
      if (data.bench_pct > 15) {
        warnings.push("⚠️ Bench élevé");
        healthStatus = 'critical';
      }
      
      if (data.alerts.attrition_risk === 'Medium' || data.alerts.attrition_risk === 'High') {
        warnings.push("⚠️ Risque d'attrition");
        if (data.alerts.attrition_risk === 'High') healthStatus = 'critical';
      }
      
      return { status: healthStatus, warnings, total };
    }

    function createBUCard(buName, data, health) {
      const card = document.createElement('div');
      card.className = 'bu-card';
      card.onclick = () => showBUDetails(buName, data, health);
      
      const headcount = data.headcount_by_grade;
      
      card.innerHTML = `
        <div class="bu-card-header">
          <div class="bu-name">${buName}</div>
          <div class="health-indicator health-${health.status}"></div>
        </div>
        <div class="pyramid-container">
          <canvas id="pyramid-${buName.replace(/\s+/g, '')}" width="260" height="140"></canvas>
        </div>
        <div class="metrics-row">
          <div class="metric">
            <div class="metric-value">${data.utilization_pct}%</div>
            <div class="metric-label">Utilisation</div>
          </div>
          <div class="metric">
            <div class="metric-value">${data.bench_pct}%</div>
            <div class="metric-label">Bench</div>
          </div>
        </div>
        ${health.warnings.length > 0 ? `
          <div style="margin-top: 12px; padding: 8px; background: rgba(251, 188, 4, 0.1); border: 1px solid rgba(251, 188, 4, 0.3); border-radius: 8px; font-size: 11px; color: #ffd166;">
            ${health.warnings.slice(0, 2).join('<br>')}
          </div>
        ` : ''}
      `;
      
      return card;
    }

    function drawPyramidChart(canvasId, data) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      
      ctx.clearRect(0, 0, width, height);
      
      const levels = [
        { name: 'Manager', count: data.Manager, color: '#9334e4' },
        { name: 'Principal', count: data.Principal, color: '#1a73e8' },
        { name: 'Senior', count: data.Senior, color: '#34a853' },
        { name: 'Junior', count: data.Junior, color: '#fbbc04' }
      ];
      
      const total = Object.values(data).reduce((sum, count) => sum + count, 0);
      const maxCount = Math.max(...Object.values(data));
      const maxWidth = width * 0.8;
      const levelHeight = height / levels.length;
      
      levels.forEach((level, index) => {
        const barWidth = (level.count / maxCount) * maxWidth;
        const x = (width - barWidth) / 2;
        const y = index * levelHeight;
        const percentage = ((level.count / total) * 100).toFixed(1);
        
        const gradient = ctx.createLinearGradient(x, y, x + barWidth, y);
        gradient.addColorStop(0, level.color);
        gradient.addColorStop(1, level.color + '80');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y + 5, barWidth, levelHeight - 10);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(level.name, x - 8, y + levelHeight / 2 + 4);
        
        ctx.textAlign = 'center';
        ctx.fillText(`${level.count} (${percentage}%)`, x + barWidth / 2, y + levelHeight / 2 + 4);
      });
    }

    function showBUDetails(buName, data, health) {
      const modal = document.createElement('div');
      modal.style.cssText = `
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.8); z-index: 2000;
        display: flex; align-items: center; justify-content: center;
        backdrop-filter: blur(8px);
      `;
      
      const content = document.createElement('div');
      content.style.cssText = `
        background: var(--bg-secondary); border: 1px solid var(--border);
        border-radius: var(--radius-lg); padding: 32px; max-width: 600px;
        width: 90vw; max-height: 80vh; overflow-y: auto;
        box-shadow: var(--shadow-lg);
      `;
      
      const headcount = data.headcount_by_grade;
      const total = Object.values(headcount).reduce((sum, count) => sum + count, 0);
      
      content.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
          <h2 style="margin: 0; font-size: 24px; color: var(--text-primary);">${buName}</h2>
          <button onclick="this.closest('[style*=fixed]').remove()" style="background: none; border: none; color: var(--text-muted); font-size: 24px; cursor: pointer;">&times;</button>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px;">
          <div>
            <h3 style="color: var(--text-primary); margin: 0 0 12px;">Effectifs par grade</h3>
            <div style="background: var(--bg-primary); padding: 16px; border-radius: var(--radius);">
              <canvas id="modal-pyramid" width="200" height="160"></canvas>
            </div>
            <div style="margin-top: 12px; font-size: 12px; color: var(--text-muted);">
              Total: ${total} consultants
            </div>
          </div>
          
          <div>
            <h3 style="color: var(--text-primary); margin: 0 0 12px;">Métriques clés</h3>
            <div style="display: grid; gap: 12px;">
              <div style="background: var(--surface); padding: 12px; border-radius: var(--radius); border: 1px solid var(--border);">
                <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);">${data.utilization_pct}%</div>
                <div style="font-size: 12px; color: var(--text-muted);">Utilisation</div>
              </div>
              <div style="background: var(--surface); padding: 12px; border-radius: var(--radius); border: 1px solid var(--border);">
                <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);">${data.bench_pct}%</div>
                <div style="font-size: 12px; color: var(--text-muted);">Bench</div>
              </div>
              <div style="background: var(--surface); padding: 12px; border-radius: var(--radius); border: 1px solid var(--border);">
                <div style="font-size: 20px; font-weight: 600; color: var(--text-primary);">${data.avg_absence_days_ytd}</div>
                <div style="font-size: 12px; color: var(--text-muted);">Jours d'absence (YTD)</div>
              </div>
            </div>
          </div>
        </div>
        
        ${health.warnings.length > 0 ? `
          <div style="background: rgba(251, 188, 4, 0.1); border: 1px solid rgba(251, 188, 4, 0.3); border-radius: var(--radius); padding: 16px; margin-bottom: 24px;">
            <h4 style="color: #ffd166; margin: 0 0 8px;">Alertes détectées</h4>
            ${health.warnings.map(warning => `<div style="color: #ffd166; font-size: 13px; margin-bottom: 4px;">${warning}</div>`).join('')}
          </div>
        ` : ''}
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
          <div>
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">TJM médians</h4>
            <div style="background: var(--surface); padding: 16px; border-radius: var(--radius); border: 1px solid var(--border);">
              ${Object.entries(data.tjm_median_by_role).map(([role, tjm]) => `
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                  <span style="color: var(--text-secondary); font-size: 13px;">${role}</span>
                  <span style="color: var(--text-primary); font-weight: 500;">${tjm}€</span>
                </div>
              `).join('')}
            </div>
          </div>
          
          <div>
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">Salaires moyens</h4>
            <div style="background: var(--surface); padding: 16px; border-radius: var(--radius); border: 1px solid var(--border);">
              ${Object.entries(data.salary_avg_by_grade).map(([grade, salary]) => `
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                  <span style="color: var(--text-secondary); font-size: 13px;">${grade}</span>
                  <span style="color: var(--text-primary); font-weight: 500;">${salary.toLocaleString()}€</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
        
        ${data.alerts.skills_gap.length > 0 ? `
          <div style="margin-top: 24px;">
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">Compétences manquantes</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
              ${data.alerts.skills_gap.map(skill => `
                <span style="background: rgba(234, 67, 53, 0.2); color: #ef476f; padding: 4px 8px; border-radius: 12px; font-size: 12px; border: 1px solid rgba(234, 67, 53, 0.3);">
                  ${skill}
                </span>
              `).join('')}
            </div>
          </div>
        ` : ''}
        
        ${data.open_offers.length > 0 ? `
          <div style="margin-top: 24px;">
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">Offres ouvertes</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
              ${data.open_offers.map(offer => `
                <span style="background: rgba(26, 115, 232, 0.2); color: var(--primary-light); padding: 4px 8px; border-radius: 12px; font-size: 12px; border: 1px solid rgba(26, 115, 232, 0.3);">
                  ${offer}
                </span>
              `).join('')}
            </div>
          </div>
        ` : ''}
      `;
      
      modal.appendChild(content);
      document.body.appendChild(modal);
      
      setTimeout(() => drawPyramidChart('modal-pyramid', headcount), 100);
      
      modal.onclick = (e) => {
        if (e.target === modal) modal.remove();
      };
    }

    window.openBUHealth = function() {
      const output = document.getElementById('out');
      output.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: var(--text-primary); margin: 0 0 16px;">Analyse de santé des Business Units</h3>
          <p style="color: var(--text-secondary); margin-bottom: 20px;">Cliquez sur une BU ci-dessus pour voir le détail de sa santé organisationnelle</p>
          
          <div style="background: var(--surface); padding: 20px; border-radius: var(--radius); margin: 20px 0; text-align: left;">
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">Indicateurs de santé</h4>
            <div style="display: grid; gap: 8px; font-size: 13px;">
              <div style="color: var(--text-secondary);">🟢 <strong>Bonne santé:</strong> Structure équilibrée, bonne utilisation, faible bench</div>
              <div style="color: var(--warning);">🟡 <strong>Attention:</strong> Déséquilibres détectés, surveillance recommandée</div>
              <div style="color: var(--accent);">🔴 <strong>Critique:</strong> Problèmes structurels majeurs, action immédiate requise</div>
            </div>
          </div>
          
          <div style="background: var(--surface); padding: 20px; border-radius: var(--radius); text-align: left;">
            <h4 style="color: var(--text-primary); margin: 0 0 12px;">Résumé global</h4>
            ${Object.entries(businessUnitsData).map(([buName, data]) => {
              const health = calculateBUHealth(data);
              const total = Object.values(data.headcount_by_grade).reduce((sum, count) => sum + count, 0);
              const statusEmoji = health.status === 'good' ? '🟢' : health.status === 'warning' ? '🟡' : '🔴';
              return `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border);">
                  <span style="color: var(--text-primary);"><strong>${buName}</strong> (${total} consultants)</span>
                  <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 12px; color: var(--text-muted);">${data.utilization_pct}% util.</span>
                    <span>${statusEmoji}</span>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
      output.classList.add('has-content');
    }

    // Main send function
    window.send = async function(textareaId) {
      const ta = document.getElementById(textareaId);
      const st = document.getElementById("st-" + textareaId.split("-")[1]);
      const out = document.getElementById("out");
      const payload = { input: ta.value.trim() };
      
      if (!payload.input) {
        ta.focus();
        return;
      }
      
      st.innerHTML = '<span class="loading">Envoi</span>';
      out.innerHTML = '<span class="loading">Traitement en cours</span>';
      out.classList.remove('has-content');
      
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        const responseText = data.text || "(réponse vide)";
        
        out.textContent = responseText;
        out.classList.add('has-content');
        st.textContent = "✓ Envoyé";
        
        addToHistory(payload.input, responseText);
        
        setTimeout(() => st.textContent = "", 2000);
        
      } catch (e) {
        out.textContent = "Erreur: " + e.message;
        out.classList.add('has-content');
        st.textContent = "❌ Erreur";
      }
    }

    window.preset = function(kind) {
      const examples = {
        offer: "J'ai une mission Data Engineer à Lyon qui commence en octobre, qui est dispo ?",
        consultant: "J'ai un consultant senior Python/Cloud basé à Paris, quelles missions sont dispo ?",
        free: "Explique-moi le principe de l'indexation vectorielle en 3 phrases."
      };
      const ids = { offer: "ta-offer", consultant: "ta-consultant", free: "ta-free" };
      
      document.getElementById(ids[kind]).value = examples[kind];
      document.getElementById(ids[kind]).focus();
    }

    setTimeout(() => {
      Object.entries(businessUnitsData).forEach(([buName, data]) => {
        const canvasId = `pyramid-${buName.replace(/\s+/g, '')}`;
        drawPyramidChart(canvasId, data.headcount_by_grade);
      });
    }, 500);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return HTMLResponse(content=_UI_HTML, status_code=200)