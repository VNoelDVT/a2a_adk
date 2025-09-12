import os, json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

_here = os.path.dirname(__file__)
_data = os.path.join(_here, "../../data/offers.json")
_offers: List[Dict[str, Any]] = json.load(open(_data, "r", encoding="utf-8"))
_offers_by_id = {o["offer_id"]: o for o in _offers}

app = FastAPI(title="offer_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"ok": True, "see": ["/openapi.json", "/.well-known/agent.json"]}

@app.get("/offers")
def list_offers():
    return {"status":"success","offers": _offers}

@app.get("/offers/{offer_id}")
def get_offer(offer_id: str):
    o = _offers_by_id.get(offer_id)
    return {"status":"success","offer": o} if o else {"status":"error","error":"unknown offer_id"}

@app.post("/offers/search")
def search_offers(payload: Dict[str, Any]):
    res=[]
    for v in _offers:
        if "role" in payload and payload["role"] and payload["role"].lower() not in v["role"].lower(): continue
        if "location" in payload and payload["location"] and payload["location"].lower() not in v["location"].lower(): continue
        if "seniority" in payload and payload["seniority"] and payload["seniority"].lower()!=v["seniority"].lower(): continue
        if "languages" in payload and payload["languages"]:
            if not set(payload["languages"]).issubset(set(v["languages"])): continue
        if "stack" in payload and payload["stack"]:
            if not set(payload["stack"]).issubset(set(v["stack"])): continue
        res.append(v)
    return {"status":"success","offers":res}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    return {"name":"offer_fake_agent","description":"Fake offers catalog","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes, description="Offers")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9101"))
    uvicorn.run(app, host="0.0.0.0", port=port)