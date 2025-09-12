import os, datetime as dt
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel

class ScheduleReq(BaseModel):
    consultant_id: str
    days: int = 7
    slots_per_day: int = 2
    timezone: str = "Europe/Paris"

app=FastAPI(title="calendar_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok":True, "see":["/openapi.json","/.well-known/agent.json"]}

@app.post("/interview/schedule")
def schedule(req: ScheduleReq):
    today = dt.date.today()
    slots=[]
    for i in range(req.days):
        d = today + dt.timedelta(days=i+2)
        for s in range(req.slots_per_day):
            h = 10 if s==0 else 15
            slots.append(f"{d.isoformat()}T{h:02d}:00:00+02:00")
    return {"status":"success","consultant_id":req.consultant_id,"slots":slots}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    return {"name":"calendar_fake_agent","description":"Interview slots","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    schema=get_openapi(title=app.title, version=app.version, routes=app.routes, description="Calendar")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9106"))
    uvicorn.run(app, host="0.0.0.0", port=port)
