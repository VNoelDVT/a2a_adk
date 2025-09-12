import os
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

SURCHARGES = {"onsite_days_per_week": 20}

class FinanceQuery(BaseModel):
    budget_tjm: int
    tjm_min: int
    constraints: Dict[str, Any] = {}

app=FastAPI(title="finance_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok":True, "see":["/openapi.json","/.well-known/agent.json"]}

@app.post("/finance/checkFit")
def check_fit(q: FinanceQuery):
    onsite = int(q.constraints.get("onsite_days_per_week", 0))
    surcharge = SURCHARGES["onsite_days_per_week"] * onsite
    margin = q.budget_tjm - q.tjm_min - surcharge
    risk = "Low" if margin>=120 else ("Medium" if margin>=60 else "High")
    return {"status":"success","margin":margin,"risk":risk,"surcharge":surcharge}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    return {"name":"finance_fake_agent","description":"Finance & margin","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    schema=get_openapi(title=app.title, version=app.version, routes=app.routes, description="Finance")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9104"))
    uvicorn.run(app, host="0.0.0.0", port=port)