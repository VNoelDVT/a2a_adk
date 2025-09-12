from typing import List, Dict, Any

def languages_pass(required: List[str], has: List[str]) -> bool:
    return set(required).issubset(set(has))

def simple_policy(required_languages: List[str], consultant_languages: List[str],
                  consultant_id: str, location: str) -> Dict[str, Any]:
    ok = languages_pass(required_languages, consultant_languages)
    # simple demo: no conflict list (all pass)
    reasons = []
    if not ok:
        reasons.append("Missing required language(s)")
    return {"status": "pass" if ok else "fail", "reasons": reasons}
