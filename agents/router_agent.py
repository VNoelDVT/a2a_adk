from typing import Any, Dict, Union
from google.adk.agents import Agent

def route_payload(payload: Union[str, Dict[str,Any]]) -> Dict[str,Any]:
    # Petite routeur heuristique ; l’orchestrateur utilisera ce résultat.
    import json, re
    obj = payload
    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
        except Exception:
            obj = {"text": payload}
    keys = set(k.lower() for k in obj.keys())
    if {"offer_id","role"}.intersection(keys) or "client_id" in keys or "must" in keys:
        return {"mode":"offer_to_candidates"}
    if "ocr_fields" in keys or {"skills","competnces","stack"}.intersection(keys):
        return {"mode":"consultant_to_offers"}
    text = (obj.get("text","") if isinstance(obj,dict) else str(obj)).lower()
    if any(k in text for k in ["offre","tjm","client","mission"]):
        return {"mode":"offer_to_candidates"}
    return {"mode":"consultant_to_offers"}

router_agent = Agent(
    name="router_agent",
    model="gemini-2.0-flash-001",
    instruction="Détermine le mode de traitement: offre→candidats ou consultant→missions.",
    tools=[route_payload],
)
