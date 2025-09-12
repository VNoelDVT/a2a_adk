# agents/offer_vectorizer_agent.py
from typing import Any, Dict
from google.adk.agents import Agent

def make_offer_vector(offer: Dict[str, Any]) -> Dict[str, Any]:
    must = offer.get("must", []) or []
    nice = offer.get("nice", []) or []
    weights = {
        "must": 0.30, "nice": 0.05, "seniority": 0.10, "languages": 0.08,
        "availability": 0.10, "distance": 0.08, "finance": 0.08, "policy": 0.04, "client_stack": 0.17
    }
    client_bonus = 0.0
    cid = (offer.get("client_id") or "")
    if cid.startswith(("CLI-BNP","CLI-SG","CLI-CARREFOUR","CLI-HERMES","CLI-NAVAL","CLI-ST")):
        client_bonus = 0.07

    return {
        "vector": {
            "must": must,
            "nice": nice,
            "seniority": offer.get("seniority"),
            "languages": offer.get("languages", []) or [],
            "start_by": offer.get("start_by"),
            "location": offer.get("location"),
            "client_id": cid,
        },
        "weights": weights,
        "client_bonus": client_bonus
    }

offer_vectorizer_agent = Agent(
    name="offer_vectorizer_agent",
    model="gemini-2.0-flash-001",
    instruction="Transforme une OfferSpec en vecteur pondéré pour le matching.",
    tools=[make_offer_vector],
)
