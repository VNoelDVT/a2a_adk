import os, json, datetime as dt
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

_here = os.path.dirname(__file__)
_cvdata = json.load(open(os.path.join(_here, "../../data/consultants_ocr.json"), "r", encoding="utf-8"))

AVAIL = {}
for c in _cvdata:
    o = c.get("OCR_fields", {})
    availability = o.get("availability") or o.get("avail") or {}
    free_from = o.get("free_from") or availability.get("from") or "2025-09-29"
    load = o.get("load_pct") or availability.get("load") or "80%"
    if isinstance(load, str) and load.endswith("%"):
        load = int(load[:-1])
    AVAIL[c["consultant_id"]] = {"from": str(free_from), "load": f"{load}%"}

app=FastAPI(title="capacity_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok":True,"see":["/openapi.json","/.well-known/agent.json"]}

@app.get("/capacity/{consultant_id}")
def capacity(consultant_id:str):
    c=AVAIL.get(consultant_id)
    return {"status":"success","availability":{"consultant_id":consultant_id, **c}} if c else {"status":"error","error":"unknown consultant_id"}

@app.post("/capacity/whatif")
def whatif(w: Dict[str, Any]):
    cid = w.get("consultant_id"); start_by = w.get("start_by"); load = int(w.get("load",80))
    avail=AVAIL.get(cid)
    if not avail: return {"status":"error","error":"unknown consultant_id"}
    a=dt.date.fromisoformat(avail["from"]); sb=dt.date.fromisoformat(start_by) if start_by else a
    cap=int(str(avail["load"]).replace("%",""))
    ok=a<=sb and load<=cap
    return {"status":"success","can_staff": ok, "reason": None if ok else "Insufficient capacity or start date too early"}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    return {"name":"capacity_fake_agent","description":"Fake capacity","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    schema=get_openapi(title=app.title, version=app.version, routes=app.routes, description="Capacity")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9103"))
    uvicorn.run(app, host="0.0.0.0", port=port)
