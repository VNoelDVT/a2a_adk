import os, json, statistics
from typing import Dict, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

_here = os.path.dirname(__file__)
_bus = json.load(open(os.path.join(_here, "../../data/business_units.json"), "r", encoding="utf-8"))
_offers = json.load(open(os.path.join(_here, "../../data/offers.json"), "r", encoding="utf-8"))

app = FastAPI(title="bu_fake_agent", version="1.0.0", openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def health():
    return {"ok": True, "see": ["/openapi.json", "/.well-known/agent.json"]}

@app.get("/bu/{name}")
def bu_health(name: str):
    bu = _bus.get(name)
    if not bu:
        return {"status": "error", "error": "unknown BU"}
    util = bu["utilization_pct"]
    bench = bu["bench_pct"]
    absd = bu["avg_absence_days_ytd"]
    head = bu["headcount_by_grade"]
    offers = [o for o in _offers if o["offer_id"] in bu.get("open_offers", [])]
    margin_hint = {}
    for role, tjm in bu.get("tjm_median_by_role", {}).items():
        # coût jour moyen ~ salaire annuel / 220 jours (très grossier)
        cost = _avg_salary(bu) / 220
        margin_hint[role] = round(tjm - cost, 1)
    recos = _make_reco(name, bu, offers)
    return {
        "status": "success",
        "bu": name,
        "metrics": {
            "utilization_pct": util,
            "bench_pct": bench,
            "avg_absence_days_ytd": absd,
            "margin_hint": margin_hint,
            "open_offers": offers,
        },
        "recommendations": recos,
    }

def _avg_salary(bu: Dict[str, Any]) -> float:
    s = bu["salary_avg_by_grade"].values()
    return statistics.mean(s) if s else 70000.0

def _make_reco(name: str, bu: Dict[str, Any], offers):
    rec = []
    util = bu["utilization_pct"]
    bench = bu["bench_pct"]
    gaps = bu.get("alerts", {}).get("skills_gap", [])
    if util < 80:
        rec.append("Accélérer le staffing sur les offres ouvertes.")
    if bench > 8:
        rec.append("Réallouer des consultants en bench vers Data/Cloud proches.")
    if gaps:
        rec.append(f"Plan de formation/certification ciblé sur {', '.join(gaps)}.")
    if name == "Cloud Platforms":
        rec.append("Renforcer OpenShift/Service Mesh (2 seniors).")
    if name == "Data & AI":
        rec.append("Former 10 profils sur FinOps/MLOps.")
    return rec

@app.get("/.well-known/agent.json")
def agent_card(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme == "https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    return {
        "name": "bu_fake_agent",
        "description": "BU health & what-if",
        "service": {"openapi_url": f"{base}/openapi.json"},
    }

@app.get("/openapi.json")
def custom_openapi(request: Request):
    host = request.headers.get("host")
    scheme = "https" if request.url.scheme == "https" or (host and host.endswith("ngrok-free.app")) else "http"
    base = f"{scheme}://{host}"
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes, description="BU Health")
    schema["openapi"] = "3.0.3"
    schema["servers"] = [{"url": base}]
    return JSONResponse(schema)

# --- Main entrypoint for standalone run ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "9109"))
    uvicorn.run(app, host="0.0.0.0", port=port)
