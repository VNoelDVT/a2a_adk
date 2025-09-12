from typing import Any, Dict, List
from google.adk.agents import Agent

def explain_shortlist(offer: Dict[str,Any], shortlist: List[Dict[str,Any]]) -> Dict[str,Any]:
    lines=[]
    for c in shortlist:
        parts=[]
        if c.get("score") is not None: parts.append(f"score={c['score']:.2f}")
        if c.get("distance_km") is not None: parts.append(f"distance≈{int(c['distance_km'])} km")
        fin=c.get("finance") or {}
        if "margin" in fin: parts.append(f"marge={fin['margin']}€ ({fin.get('risk')})")
        pol=c.get("policy") or {}
        if "status" in pol: parts.append(f"policy={pol['status']}")
        cap=c.get("capacity") or {}
        if "can_staff" in cap: parts.append("capacité=OK" if cap["can_staff"] else "capacité=KO")
        lines.append(f"- {c['consultant_id']} • " + " • ".join(parts))
    md = f"**Client:** {offer.get('client_id')}  \n**Role:** {offer.get('role')}  \n**Lieu:** {offer.get('location')}  \n\n**Raisons par candidat**\n" + "\n".join(lines)
    return {"status":"success","reasons_markdown": md}

explainability_agent = Agent(
    name="explainability_agent",
    model="gemini-2.0-flash-001",
    instruction="Génère un résumé C-level et des raisons lisibles pour une shortlist.",
    tools=[explain_shortlist],
)
