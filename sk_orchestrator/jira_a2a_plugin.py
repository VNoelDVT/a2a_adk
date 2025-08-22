# # sk_orchestrator/jira_a2a_plugin.py 
# import os, requests, re
# from urllib.parse import urlparse

# def _discover_openapi_url():
#     card_url = os.getenv("JIRA_CARD_URL", "http://127.0.0.1:9003/.well-known/agent.json")
#     card = requests.get(card_url, timeout=5).json()
#     svc = card.get("service", {})
#     openapi = svc.get("openapi_url") or (svc.get("base_url", "").rstrip("/") + "/openapi.json")
#     if not openapi:
#         raise RuntimeError("Agent Card invalide (pas d'openapi_url/base_url)")
#     return openapi

# def _api_base():
#     up = urlparse(_discover_openapi_url())
#     return f"{up.scheme}://{up.netloc}"

# def _canon(p: str) -> str:
#     return re.sub(r"[^A-Za-z0-9_]", "_", p)

# def find_risks(project: str):
#     url = _api_base() + "/findRisks"
#     return requests.post(url, json={"project": _canon(project)}, timeout=10).json()

# def create_ticket(project: str, summary: str, description: str, assignees: list,
#                   link_blocker: str | None = None, simulateFailure: bool=False):
#     url = _api_base() + "/createTicket"
#     payload = {
#         "project": _canon(project),
#         "summary": summary,
#         "description": description,
#         "assignees": assignees,
#         "link_blocker": link_blocker,
#         "simulateFailure": simulateFailure,
#     }
#     return requests.post(url, json=payload, timeout=10).json()

# def get_logs(transactionId: str):
#     url = _api_base() + "/getTicketLogs"
#     return requests.post(url, json={"transactionId": transactionId}, timeout=10).json()


import os, requests, re
from urllib.parse import urlparse



def _discover_openapi_url():
    card_url = os.getenv("JIRA_CARD_URL", "http://127.0.0.1:9003/.well-known/agent.json")
    card = requests.get(card_url, timeout=5).json()
    svc = card.get("service", {})
    openapi = svc.get("openapi_url") or (svc.get("base_url", "").rstrip("/") + "/openapi.json")
    if not openapi:
        raise RuntimeError("Agent Card invalide (openapi_url/base_url manquant).")
    return openapi

def _api_base():
    up = urlparse(_discover_openapi_url())
    return f"{up.scheme}://{up.netloc}"

def _canon(p: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", p)

def find_risks(project: str, stalled_days: int = 5):
    """Appelle l’endpoint des risques (corps JSON)."""
    url = _api_base() + "/findRisks"
    payload = {"project": _canon(project), "stalled_days": stalled_days}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def create_ticket(project: str, summary: str, description: str, assignees: list,
                  link_blocker: str | None = None, simulateFailure: bool = False):
    """Crée un ticket et renvoie transactionId/issue (si succès)."""
    url = _api_base() + "/createTicket"
    payload = {
        "project": _canon(project),
        "summary": summary,
        "description": description,
        "assignees": assignees,
        "link_blocker": link_blocker,
        "simulateFailure": simulateFailure,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def get_logs(transactionId: str):
    """Récupère les logs par transactionId."""
    url = _api_base() + "/getTicketLogs"
    payload = {"transactionId": transactionId}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()
