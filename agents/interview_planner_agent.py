import os, requests
from urllib.parse import urlparse
from typing import Any, Dict, List, Set, Tuple
from google.adk.agents import Agent

def _discover(card_env:str, default_card:str):
    card=requests.get(os.getenv(card_env,default_card),timeout=5).json()
    svc=card.get("service",{})
    openapi=svc.get("openapi_url") or (svc.get("base_url","").rstrip("/")+"/openapi.json" if svc.get("base_url") else None)
    if not openapi: raise RuntimeError(f"[{card_env}] bad card")
    spec=requests.get(openapi,timeout=5).json()
    base=f"{urlparse(openapi).scheme}://{urlparse(openapi).netloc}"
    return base,spec

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
    for path,methods in (spec.get("paths") or {}).items():
        for m,op in (methods or {}).items():
            props=set()
            for _,media in ((op.get("requestBody") or {}).get("content") or {}).items():
                props|=_schema_keys((media or {}).get("schema") or {})
            for p in (op.get("parameters") or []):
                if p.get("in") in ("query","path") and "name" in p: props.add(p["name"])
                props|=_schema_keys(p.get("schema") or {})
            score=10*len(required & props)+(50 if required.issubset(props) else 0)+(2 if m.upper()=="POST" else 0)
            if score>0: best=max(best,(score,m.upper(),f"{base}{path}"),key=lambda x:x[0])
    if not best: raise RuntimeError("No op matches.")
    return best[1],best[2]

_BASE,_SPEC=_discover("CAL_CARD_URL","http://127.0.0.1:9106/.well-known/agent.json")

def propose_slots(shortlist: List[Dict[str,Any]], days:int=7, slots_per_day:int=2, timezone:str="Europe/Paris") -> Dict[str,Any]:
    m,u=_resolve(_SPEC,_BASE,{"consultant_id","days","slots_per_day","timezone"})
    out=[]
    for c in shortlist:
        r=requests.request(m,u,json={"consultant_id":c["consultant_id"],"days":days,"slots_per_day":slots_per_day,"timezone":timezone},timeout=10).json()
        out.append({"consultant_id":c["consultant_id"],"slots":r.get("slots",[])})
    return {"status":"success","interviews":out}

interview_planner_agent = Agent(
    name="interview_planner_agent",
    model="gemini-2.0-flash-001",
    instruction="Propose des créneaux d'entretien pour chaque candidat retenu.",
    tools=[propose_slots],
)
