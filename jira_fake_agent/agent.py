import os, uuid, time
from typing import List, Optional, Dict, Any
from google.adk.agents import Agent

# -----------------------------
# Base de données "fake" en mémoire
# -----------------------------
# Tickets simplifiés pour le projet Phoenix-V2
# Champs: key, summary, status, priority, updated_days_ago, assignee, links, comments_count
TICKETS: Dict[str, Dict[str, Any]] = {
    "PHX-109": {
        "project": "Phoenix-V2",
        "summary": "Deploy to staging",
        "status": "In Progress",
        "priority": "Medium",
        "updated_days_ago": 6,            # → considéré "stalled"
        "assignee": "david.miller",
        "links": {"is_blocked_by": ["PHX-108"]},
        "comments_count": 2,
    },
    "PHX-108": {
        "project": "Phoenix-V2",
        "summary": "Fix CI pipeline flaky test",
        "status": "To Do",
        "priority": "High",
        "updated_days_ago": 2,
        "assignee": "ci.bot",
        "links": {},
        "comments_count": 14,             # → signal de controverse/confusion
    },
    "PHX-113": {
        "project": "Phoenix-V2",
        "summary": "Implement new payment gateway",
        "status": "In Progress",
        "priority": "Medium",
        "updated_days_ago": 6,            # → stalled
        "assignee": "david.miller",
        "links": {},
        "comments_count": 5,
    },
    "PHX-125": {
        "project": "Phoenix-V2",
        "summary": "Fix security vulnerability CVE-2025-1234",
        "status": "To Do",
        "priority": "Highest",            # → high priority non done
        "updated_days_ago": 1,
        "assignee": "sec.team",
        "links": {},
        "comments_count": 1,
    },
    "PHX-108B": {
        "project": "Phoenix-V2",
        "summary": "DB index tuning",
        "status": "In Progress",
        "priority": "Low",
        "updated_days_ago": 1,
        "assignee": "dba.alex",
        "links": {},
        "comments_count": 0,
    },
}

# Compteur pour créer de nouveaux tickets
_counter = 200
def _next_issue_key(project: str) -> str:
    global _counter
    _counter += 1
    prefix = "PHX" if project.lower().startswith("phoenix") else "GEN"
    return f"{prefix}-{_counter}"

# "Journal" de jobs par transactionId (pour getTicketLogs)
JOBS: Dict[str, Dict[str, Any]] = {}  # txid -> {action,status,error,result,ts,payload}


# -----------------------------
# Tools exposés par l'agent
# -----------------------------
def findRisks(project: str) -> dict:
    """
    Analyse "fake" des risques du projet.
    - stalled: In Progress & updated_days_ago >= 5
    - high: priority Highest & status != Done
    - blockers: liens is_blocked_by
    - controversial: comments_count élevé (>=10)
    """
    # filtre par projet
    items = [ (k,v) for k,v in TICKETS.items() if v.get("project") == project ]

    stalled = []
    high = []
    blockers = []
    controversial = []

    for k, v in items:
        if v["status"] == "In Progress" and v["updated_days_ago"] >= 5:
            stalled.append({"key": k, **{x:v[x] for x in ("summary","assignee","updated_days_ago")}})
        if v["priority"] == "Highest" and v["status"] != "Done":
            high.append({"key": k, **{x:v[x] for x in ("summary","status")}})
        if v.get("links", {}).get("is_blocked_by"):
            blockers.append({"key": k, "blocked_by": v["links"]["is_blocked_by"], "summary": v["summary"]})
        if v.get("comments_count", 0) >= 10:
            controversial.append({"key": k, "comments": v["comments_count"], "summary": v["summary"]})

    return {
        "status": "success",
        "project": project,
        "stalled": stalled,
        "high_priority": high,
        "blockers": blockers,
        "controversial": controversial,
    }


def createTicket(
    project: str,
    summary: str,
    description: str,
    assignees: List[str],
    link_blocker: Optional[str] = None,
    simulateFailure: bool = False
) -> dict:
    """
    Crée un ticket "fake" et journalise l'opération.
    - simulateFailure=True pour simuler un 403/échec
    """
    txid = str(uuid.uuid4())
    now = time.time()
    payload = {
        "project": project, "summary": summary, "description": description,
        "assignees": assignees, "link_blocker": link_blocker, "simulateFailure": simulateFailure
    }

    if simulateFailure:
        JOBS[txid] = {
            "action": "createTicket", "status": "Failed",
            "error": "Simulated 403 Forbidden (demo). Missing 'Create Issue' permission.",
            "result": None, "ts": now, "payload": payload
        }
        return {"status": "failed", "transactionId": txid, "error": JOBS[txid]["error"]}

    issue_key = _next_issue_key(project)
    # crée l'objet
    TICKETS[issue_key] = {
        "project": project, "summary": summary, "status": "To Do",
        "priority": "High", "updated_days_ago": 0,
        "assignee": assignees[0] if assignees else None,
        "links": {"is_blocked_by": [link_blocker]} if link_blocker else {},
        "comments_count": 0,
    }
    JOBS[txid] = {
        "action": "createTicket", "status": "Succeeded",
        "error": None, "result": {"issue": issue_key}, "ts": now, "payload": payload
    }
    return {"status": "success", "transactionId": txid, "issue": issue_key}


def getTicketLogs(transactionId: str) -> dict:
    """Retourne le log/detail d'un job par transactionId."""
    job = JOBS.get(transactionId)
    if not job:
        return {"status": "error", "error": f"Unknown transactionId {transactionId}"}
    return {
        "status": "success",
        "transactionId": transactionId,
        "job": job
    }


# -----------------------------
# Agent ADK
# -----------------------------
root_agent = Agent(
    name="jira_fake_agent",
    model="gemini-2.0-flash-001",
    description="Agent 'fake Jira' pour démo risques/projets (findRisks, createTicket, getTicketLogs).",
    instruction=(
        "Réponds en français. Utilise findRisks(project), createTicket(...), getTicketLogs(transactionId). "
        "Respecte les paramètres donnés et renvoie des réponses concises."
    ),
    tools=[findRisks, createTicket, getTicketLogs],
)
