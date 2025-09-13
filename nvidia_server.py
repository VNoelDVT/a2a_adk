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
        "N’affiche JAMAIS de balises <think> ni d’analyse interne. "
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

    # explicit city override from the prompt (e.g., “Massy”, “Lyon”, …)
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

# -------------------------- Minimal, pretty UI --------------------------
_UI_HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Orchestrateur</title>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:#0b1020; --card:#111a33; --ink:#e9eefc; --muted:#98a3c7;
      --accent:#5b8cff; --accent-2:#7dd3fc; --ring:#2640a4;
      --ok:#3ddc97; --warn:#ffd166; --err:#ef476f;
      --radius:14px; --shadow: 0 10px 30px rgba(0,0,0,.25);
    }
    * { box-sizing: border-box; }
    html,body { height:100%; }
    body {
      margin:0; font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial;
      color:var(--ink); background: radial-gradient(1200px 800px at 15% -10%, #1a2650 0%, transparent 60%),
                         radial-gradient(1200px 800px at 85% -10%, #0f1b44 0%, transparent 60%),
                         var(--bg);
    }
    .wrap { max-width:1100px; margin:40px auto; padding:0 20px; }
    h1 { font-size:28px; letter-spacing:.3px; margin:0 0 18px; }
    .chips { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
    .chip { padding:8px 12px; border:1px solid #2a3765; color:var(--ink);
            background:rgba(255,255,255,.03); border-radius:999px; cursor:pointer; user-select:none; }
    .chip:hover { border-color:var(--accent); box-shadow:0 0 0 3px rgba(91,140,255,.2) inset; }

    .grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:18px; }
    @media (max-width: 980px) { .grid { grid-template-columns:1fr; } }

    .card { background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
            border:1px solid #223061; border-radius:var(--radius); padding:16px; box-shadow:var(--shadow); }
    .card h2 { font-size:18px; margin:4px 0 12px; }
    .hint { color:var(--muted); font-size:12px; margin-top:6px; }

    textarea {
      width:100%; height:120px; resize:vertical;
      color:var(--ink); background:#0d1329; border:1px solid #283770;
      border-radius:12px; padding:12px 14px; outline:none;
    }
    textarea:focus { border-color:var(--accent); box-shadow:0 0 0 4px rgba(91,140,255,.2); }

    .row { display:flex; gap:10px; align-items:center; margin-top:10px; }
    button {
      padding:10px 14px; border-radius:12px; border:1px solid #2a3765;
      background:linear-gradient(180deg, #1a2a63, #14204d); color:var(--ink); cursor:pointer;
    }
    button:hover { border-color:var(--accent); box-shadow:0 0 0 4px rgba(91,140,255,.2); }

    .resp {
      white-space:pre-wrap; background:#0b1126; border:1px solid #283770;
      border-radius:14px; padding:16px; margin-top:16px; min-height:90px;
    }
    .muted { color:var(--muted); }
    .badge { font-size:11px; padding:3px 8px; border-radius:999px; border:1px solid #2a3765; color:#cfe2ff; }

    .footer { margin-top:28px; color:#9fb1ff; font-size:12px; opacity:.8; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Orchestrateur</h1>
    <div class="chips">
      <div class="chip" onclick="preset('offer')">Mission → Candidats</div>
      <div class="chip" onclick="preset('consultant')">Consultant → Missions</div>
      <div class="chip" onclick="preset('free')">Chat libre</div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Mission → Candidats <span class="badge">Use case #1</span></h2>
        <textarea id="ta-offer" placeholder="Ex: J’ai une mission Data Engineer à Lyon qui commence en octobre, qui est dispo ?"></textarea>
        <div class="row">
          <button onclick="send('ta-offer')">Envoyer</button>
          <span id="st-offer" class="hint muted"></span>
        </div>
      </div>

      <div class="card">
        <h2>Consultant → Missions <span class="badge">Use case #2</span></h2>
        <textarea id="ta-consultant" placeholder="Ex: J’ai un consultant senior Python/Cloud basé à Paris, quelles missions sont dispo ?"></textarea>
        <div class="row">
          <button onclick="send('ta-consultant')">Envoyer</button>
          <span id="st-consultant" class="hint muted"></span>
        </div>
      </div>

      <div class="card">
        <h2>Chat libre <span class="badge">Fallback assistant</span></h2>
        <textarea id="ta-free" placeholder="Posez n’importe quelle question… (si ce n’est pas l’un des 2 cas d’usage, l’assistant répond normalement)"></textarea>
        <div class="row">
          <button onclick="send('ta-free')">Envoyer</button>
          <span id="st-free" class="hint muted"></span>
        </div>
        <div class="hint">Astuce : collez vos prompts longs ici. L’agent détecte automatiquement le meilleur traitement.</div>
      </div>
    </div>

    <div class="card" style="margin-top:18px;">
      <h2>Réponse</h2>
      <div id="out" class="resp muted">La réponse apparaîtra ici…</div>
    </div>

    <div class="footer">POST <code>/chat</code> · JSON { "input": "…" } — L’interface détecte l’intention et route vers le bon flux. Hors cas d’usage, elle agit comme un chatbot.</div>
  </div>

  <script>
    function preset(kind){
      const examples = {
        offer: "J’ai une mission Data Engineer à Lyon qui commence en octobre, qui est dispo ?",
        consultant: "J’ai un consultant senior Python/Cloud basé à Paris, quelles missions sont dispo ?",
        free: "Explique-moi le principe de l’indexation vectorielle en 3 phrases."
      };
      const ids = {offer:"ta-offer", consultant:"ta-consultant", free:"ta-free"};
      document.getElementById(ids[kind]).value = examples[kind];
      document.getElementById(ids[kind]).focus();
    }

    async function send(textareaId){
      const ta = document.getElementById(textareaId);
      const st = document.getElementById("st-" + textareaId.split("-")[1]);
      const out = document.getElementById("out");
      const payload = { input: ta.value.trim() };
      if(!payload.input){ ta.focus(); return; }
      st.textContent = "Envoi…";
      out.textContent = "⏳ Traitement en cours…";
      out.classList.add("muted");
      try{
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        out.textContent = data.text || "(réponse vide)";
        out.classList.remove("muted");
        st.textContent = "OK";
        setTimeout(()=> st.textContent="", 1200);
      }catch(e){
        out.textContent = "Erreur: " + e;
        out.classList.remove("muted");
        st.textContent = "Erreur";
      }
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def ui_root():
    return HTMLResponse(content=_UI_HTML, status_code=200)
