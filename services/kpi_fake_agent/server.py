import os, time
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

LOGS: List[Dict[str, Any]] = []

class KPIEntry(BaseModel):
    step: str
    duration_ms: int
    tool_calls: int
    fit_avg: float | None = None
    margin_avg: float | None = None
    notes: str | None = None

app=FastAPI(title="kpi_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok":True, "see":["/openapi.json","/.well-known/agent.json"]}

@app.post("/kpi/log")
def kpi_log(e: KPIEntry):
    payload = e.dict() | {"ts": time.time()}
    LOGS.append(payload)
    return {"status":"success","stored":payload}

@app.get("/kpi")
def kpi_all():
    return {"status":"success","logs":LOGS, "count": len(LOGS)}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    return {"name":"kpi_fake_agent","description":"KPI logger","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    schema=get_openapi(title=app.title, version=app.version, routes=app.routes, description="KPI")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9108"))
    uvicorn.run(app, host="0.0.0.0", port=port)