import os, json, datetime as dt
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from libs.common.normalizer import normalize_consultant_ocr

# ---------------------------
# Load fake CV data
# ---------------------------
_here = os.path.dirname(__file__)
_data = os.path.join(_here, "../../data/consultants_ocr.json")

with open(_data, "r", encoding="utf-8") as f:
    _cv_raw: List[Dict[str, Any]] = json.load(f)

_cvs = [normalize_consultant_ocr(c) for c in _cv_raw]
_by_id = {c["consultant_id"]: c for c in _cvs}

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="cv_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"ok": True, "see": ["/openapi.json", "/.well-known/agent.json"]}

@app.get("/cvs/{consultant_id}")
def get_cv(consultant_id: str):
    cv = _by_id.get(consultant_id)
    return {"status": "success", "cv": cv} if cv else {"status": "error", "error": "unknown consultant_id"}

@app.post("/cvs/search")
def search_cvs(q: Dict[str, Any]):
    must = q.get("must") or []
    nice = q.get("nice") or []
    seniority = q.get("seniority")
    languages = q.get("languages") or []
    start_by = q.get("start_by")

    res = []
    for cv in _cvs:
        s = 0.0
        s += 0.3 * sum(1 for k in must if k in cv["skills"])
        s += 0.05 * sum(1 for k in nice if k in cv["skills"])
        if seniority and cv["grade"].lower() == seniority.lower():
            s += 0.1
        if languages and set(languages).issubset(set(cv["languages"])):
            s += 0.08
        if start_by:
            try:
                sb = dt.date.fromisoformat(start_by)
                # simplification: accept always (for demo)
                s += 0.1
            except Exception:
                pass
        res.append({
            "consultant_id": cv["consultant_id"],
            "name_masked": cv["name_masked"],
            "grade": cv["grade"],
            "skills": cv["skills"],
            "languages": cv["languages"],
            "city": cv["city"],
            "tjm_min": cv["tjm_min"],
            "availability": cv["availability"],
            "score": round(min(s, 0.99), 2)
        })
    res.sort(key=lambda x: x["score"], reverse=True)
    return {"status": "success", "results": res}

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme == "https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    return {
        "name": "cv_fake_agent",
        "description": "Fake CV catalog",
        "service": {"openapi_url": f"{base}/openapi.json"}
    }

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme == "https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes, description="CVs")
    schema["openapi"] = "3.0.3"
    schema["servers"] = [{"url": base}]
    return JSONResponse(schema)

# ---------------------------
# Entrypoint for uvicorn
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9102"))
    uvicorn.run(app, host="0.0.0.0", port=port)
