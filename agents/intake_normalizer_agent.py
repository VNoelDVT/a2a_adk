import re
from typing import Any, Dict
from google.adk.agents import Agent

OfferSpec = Dict[str, Any]
ConsultantSpec = Dict[str, Any]

# --- Simplified demo normalizer ---
def detect_and_normalize(payload: str) -> Dict[str, Any]:
    """
    Demo-only version of intake normalizer.
    Covers 3 scenarios:
    - Offer -> Candidates
    - Consultant -> Offers
    - BU -> Health report
    """

    text = payload.lower()

    # Case 1: BU health
    if "bu" in text or "rapport de santé" in text:
        m = re.search(r"bu\s+([a-zA-Z& ]+)", text)
        bu_name = m.group(1).strip() if m else "Data & AI"
        return {"mode": "bu_report", "bu_name": bu_name}

    # Case 2: consultant asking for offers
    if any(k in text for k in ["consultant", "cv", "profil"]):
        return {
            "mode": "consultant_to_offers",
            "consultant": {
                "consultant_id": "CONS-001",
                "name_masked": "Consultant-XXX",
                "skills": ["Python", "Cloud"],
                "location": "Paris",
                "grade": "Senior",
            },
        }

    # Case 3: offer to candidates
    if any(k in text for k in ["mission", "offre", "project"]):
        return {
            "mode": "offer_to_candidates",
            "offer": {
                "offer_id": "OFF-001",
                "client_id": "Société Générale",
                "role": "Data Engineer",
                "stack": ["Python", "Airflow", "GCP"],
                "location": "Lyon",
                "budget_tjm": 750,
                "start_by": "2025-10-01",
            },
        }

    # Default fallback
    return {"mode": "unknown", "raw": payload}

# --- ADK Agent wrapper ---
intake_normalizer_agent = Agent(
    name="intake_normalizer_agent",
    model="gemini-2.0-flash-001",
    instruction="Détecte si l'entrée correspond à une OFFRE, un CONSULTANT ou une BU (rapport santé).",
    tools=[detect_and_normalize],
)
