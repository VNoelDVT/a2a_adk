import os, re, requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set, Tuple
from google.adk.agents import Agent

def _discover(env_var="JIRA_CARD_URL", default="http://127.0.0.1:9107/.well-known/agent.json"):
    card_url = os.getenv(env_var, default)
    card = requests.get(card_url, timeout=5).json()

    # ⚡ Tolérer les deux formats de card
    svc = card.get("service", card)  
    openapi = svc.get("openapi_url")

    if not openapi:
        raise RuntimeError(f"Agent Card invalide pour JIRA: {card}")

    return (os.path.dirname(openapi), requests.get(openapi, timeout=5).json())

def _schema_keys(s):
    ks=set()
    def w(x):
        if not isinstance(x,dict): return
        for k,v in (x.get("properties") or {}).items(): ks.add(k); w(v)
        if "items" in x: w(x["items"])
        for alt in ("anyOf","oneOf","allOf"):
            for ss in (x.get(alt) or []): w(ss)
    w(s or {}); return ks

def _resolve(spec, base, required:Set[str]):
    best=None
    for p,ms in (spec.get("paths") or {}).items():
        for m,op in (ms or {}).items():
            props=set()
            for _,media in ((op.get("requestBody") or {}).get("content") or {}).items(): props|=_schema_keys((media or {}).get("schema") or {})
            for prm in (op.get("parameters") or []):
                if prm.get("in") in ("query","path") and "name" in prm: props.add(prm["name"])
                props|=_schema_keys(prm.get("schema") or {})
            sc=10*len(required & props)+(50 if required.issubset(props) else 0)+(2 if m.upper()=="POST" else 0)
            if sc>0: best=max(best,(sc,m.upper(),f"{base}{p}"),key=lambda x:x[0])
    if not best: raise RuntimeError("No op matches for Jira.")
    return best[1],best[2]

_BASE,_SPEC=_discover()

def create_tickets_for_shortlist(project: str, offer: Dict[str,Any], shortlist: List[Dict[str,Any]], simulate_failure_for: Optional[str]=None) -> Dict[str,Any]:
    m,u=_resolve(_SPEC,_BASE,{"project","summary","description","assignees"})
    created=[]
    for c in shortlist:
        desc=f"Client: {offer.get('client_id')} | Role: {offer.get('role')} | Budget TJM: {offer.get('budget_tjm')} | Location: {offer.get('location')}"
        payload={"project":project,"summary":f"Entretien {offer.get('role')} - {c['consultant_id']}",
                 "description":desc,"assignees":[c['consultant_id']],
                 "link_blocker": None, "simulateFailure": (simulate_failure_for==c['consultant_id'])}
        r=requests.request(m,u,json=payload,timeout=10).json()
        created.append({"consultant_id":c["consultant_id"], **r})
    return {"status":"success","tickets":created}

jira_actions_agent = Agent(
    name="jira_actions_agent",
    model="gemini-2.0-flash-001",
    instruction="Crée un ticket Jira par candidat sélectionné.",
    tools=[create_tickets_for_shortlist],
)
