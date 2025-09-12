import os
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from libs.common.policy_rules import simple_policy

class PolicyQuery(BaseModel):
    required_languages: List[str]
    consultant_languages: List[str]
    consultant_id: str
    location: str

app=FastAPI(title="policy_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health(): return {"ok":True, "see":["/openapi.json","/.well-known/agent.json"]}

@app.post("/policy/check")
def check_policy(q: PolicyQuery):
    return {"status":"success", **simple_policy(q.required_languages, q.consultant_languages, q.consultant_id, q.location)}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    return {"name":"policy_fake_agent","description":"Policy & compliance","service":{"openapi_url": f"{base}/openapi.json"}}

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host=request.headers.get("host"); scheme="https" if request.url.scheme=="https" or (host and host.endswith("ngrok-free.app")) else "http"
    base=f"{scheme}://{host}"
    schema=get_openapi(title=app.title, version=app.version, routes=app.routes, description="Policy")
    schema["openapi"]="3.0.3"; schema["servers"]=[{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9105"))
    uvicorn.run(app, host="0.0.0.0", port=port)