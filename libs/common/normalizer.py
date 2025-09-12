from __future__ import annotations
from typing import Dict, Any
import re

def _as_pct(x):
    if isinstance(x, str) and x.endswith("%"):
        try:
            return int(x[:-1])
        except:
            return 80
    if isinstance(x, (int, float)):
        return int(x)
    return 80

def normalize_consultant_ocr(entry: Dict[str, Any]) -> Dict[str, Any]:
    ocr = entry.get("OCR_fields", {}) or {}
    skills = (
        ocr.get("competnces")
        or ocr.get("skills")
        or ocr.get("stack")
        or []
    )
    # canonicalize some skill keys
    norm = [re.sub(r"\bPy\b", "Python", s, flags=re.I) for s in skills]
    norm = [re.sub(r"\bAir\s*Flow\b", "Airflow", s, flags=re.I) for s in norm]

    languages = ocr.get("lang") or ocr.get("languages") or ocr.get("langues") or []
    city = ocr.get("loc") or ocr.get("city") or ocr.get("base") or "Paris"
    tjm = ocr.get("tjm_min") or ocr.get("min_rate") or ocr.get("tjm_floor") or 650

    # availability variants
    avail = ocr.get("avail") or ocr.get("availability") or {}
    free_from = ocr.get("free_from") or avail.get("from") or "2025-09-29"
    load_pct = ocr.get("load_pct") or _as_pct(avail.get("load", 80))

    years = ocr.get("yrs_exp") or ocr.get("exp_years") or 7

    return {
        "consultant_id": entry.get("consultant_id"),
        "name_masked": entry.get("full_name_masked") or entry.get("name_masked"),
        "grade": entry.get("grade") or "Senior",
        "skills": norm,
        "languages": languages,
        "city": city,
        "tjm_min": int(tjm),
        "availability": {"from": str(free_from), "load_pct": int(load_pct)},
        "years_experience": int(years),
    }

def normalize_offer_like(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Support texte/clé partiel -> OfferSpec minimal
    out = {}
    for k in ("offer_id","client_id","role","stack","seniority","languages","location",
              "budget_tjm","start_by","must","nice","constraints","urgency"):
        if k in payload:
            out[k] = payload[k]
    # heuristiques simples
    if "stack" not in out and "must" in out:
        out["stack"] = list(set(out["must"] + out.get("nice", [])))
    if "constraints" not in out:
        out["constraints"] = {"onsite_days_per_week": 2, "eu_work_permit": True}
    return out
