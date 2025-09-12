import os, requests
from urllib.parse import urlparse
from typing import Any, Dict, Set, Tuple
from google.adk.agents import Agent

def _discover():
    card_url=os.getenv("BU_CARD_URL","http://127.0.0.1:9108/.well-known/agent.json")
    card=requests.get(card_url,timeout=5).json()
    svc=card.get("service",{})
    openapi=svc.get("openapi_url") or (svc.get("base_url","").rstrip("/")+"/openapi.json" if svc.get("base_url") else None)
    if not openapi: raise RuntimeError("Agent Card invalide pour BU.")
    spec=requests.get(openapi,timeout=5).json()
    up=urlparse(openapi); base=f"{up.scheme}://{up.netloc}"
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
    for p,ms in (spec.get("paths") or {}).items():
        for m,op in (ms or {}).items():
            props=set()
            for _,media in ((op.get("requestBody") or {}).get("content") or {}).items(): props|=_schema_keys((media or {}).get("schema") or {})
            for prm in (op.get("parameters") or []):
                if prm.get("in") in ("query","path") and "name" in prm: props.add(prm["name"])
                props|=_schema_keys(prm.get("schema") or {})
            sc=10*len(required & props)+(50 if required.issubset(props) else 0)+(2 if m.upper()=="POST" else 0)
            if sc>0: best=max(best,(sc,m.upper(),f"{base}{p}"), key=lambda x:x[0])
    if not best: raise RuntimeError("No op matches for BU.")
    return best[1],best[2]

_BASE,_SPEC=_discover()

def bu_overview(bu_name: str) -> Dict[str,Any]:
    m,u=_resolve(_SPEC,_BASE,{"bu_name"})
    return requests.request(m,u,json={"bu_name":bu_name},timeout=10).json()

def bu_whatif(bu_name:str, hire_delta:int=0, train_focus:str="") -> Dict[str,Any]:
    m,u=_resolve(_SPEC,_BASE,{"bu_name","hire_delta","train_focus"})
    return requests.request(m,u,json={"bu_name":bu_name,"hire_delta":hire_delta,"train_focus":train_focus},timeout=10).json()

bu_health_agent = Agent(
    name="bu_health_agent",
    model="gemini-2.0-flash-001",
    instruction="Analyse la santé d'une BU et propose des mesures (what-if).",
    tools=[bu_overview, bu_whatif],
)
