# from google.adk.agents import Agent
# from time_agent.agent import get_current_time
# from weather_agent.agent import get_weather

# def ask_time(city: str) -> dict:
#     """Route vers le sous-agent temps."""
#     return get_current_time(city)

# def ask_weather(city: str) -> dict:
#     """Route vers le sous-agent météo."""
#     return get_weather(city)

# root_agent = Agent(
#     name="orchestrator_agent",
#     model="gemini-2.0-flash-001",
#     description="Orchestrateur qui route vers time_agent ou weather_agent.",
#     instruction=(
#         "Réponds en français. Choisis l'outil adapté en fonction de la question. "
#         "Utilise ask_time pour l'heure et ask_weather pour la météo. "
#         "Si la ville n'est pas supportée, explique-le clairement."
#     ),
#     tools=[ask_time, ask_weather],
# )

# 

# from typing import List, Optional

# from google.adk.agents import Agent

# # --- Essai n°1 : via OpenAPI (A2A réseau) ---
# JIRA_FAKE_OPENAPI = "http://127.0.0.1:9002/openapi.json"
# _tools = None
# _reason = None

# try:
#     # Selon les versions d'ADK, le chemin du tool change :
#     try:
#         from google.adk.tools.openapi_tool import OpenAPITool  # ADK récent
#     except Exception:
#         from google.adk.tools.open_api_tool import OpenAPITool  # ADK plus ancien

#     jira_tool = OpenAPITool.from_url(
#         JIRA_FAKE_OPENAPI,
#         name="jira_fake",
#         description="Outils de l'agent Jira factice: findRisks, createTicket, getTicketLogs."
#     )
#     _tools = [jira_tool]

# except Exception as e:
#     # --- Fallback : in-process (A2A léger) ---
#     _reason = f"OpenAPITool indisponible: {e}. Fallback in-process."
#     from jira_fake_agent.agent import findRisks, createTicket, getTicketLogs

#     def jira_findRisks(project: str) -> dict:
#         """Analyse les risques d'un projet (stalled, high priority, blockers, controversial)."""
#         return findRisks(project)

#     def jira_createTicket(
#         project: str,
#         summary: str,
#         description: str,
#         assignees: list[str],
#         link_blocker: Optional[str] = None,
#         simulateFailure: bool = False,
#     ) -> dict:
#         return createTicket(project, summary, description, assignees, link_blocker, simulateFailure)

#     def jira_getTicketLogs(transactionId: str) -> dict:
#         """Retourne les logs détaillés d'un job par transactionId."""
#         return getTicketLogs(transactionId)

#     _tools = [jira_findRisks, jira_createTicket, jira_getTicketLogs]

# # --- Agent orchestrateur ---
# root_agent = Agent(
#     name="orchestrator_agent",
#     model="gemini-2.0-flash-001",
#     description="Orchestrateur A2A pour gestion de risques projet.",
#     instruction=(
#         "Réponds en français. "
#         "Si disponible, utilise le tool OpenAPI 'jira_fake' (findRisks, createTicket, getTicketLogs). "
#         "Sinon, utilise les outils locaux du fallback aux mêmes noms et signatures. "
#         "Pour les actions longues ou ayant échoué, renvoie le transactionId et explique comment obtenir les logs."
#         + (f"  NOTE TECHNIQUE: {_reason}" if _reason else "")
#     ),
#     tools=_tools,
# )

# import requests
# from typing import List, Optional
# from google.adk.agents import Agent

# BASE = "http://127.0.0.1:9002"  # api_server du fake Jira

# def _resolve_op(operation_id: str):
#     spec = requests.get(f"{BASE}/openapi.json", timeout=5).json()
#     for path, methods in spec["paths"].items():
#         for method, op in methods.items():
#             if op.get("operationId") == operation_id:
#                 return method.upper(), f"{BASE}{path}"
#     raise RuntimeError(f"operationId '{operation_id}' introuvable")

# def jira_findRisks(project: str) -> dict:
#     m, u = _resolve_op("findRisks")
#     return requests.request(m, u, json={"project": project}, timeout=10).json()

# def jira_createTicket(project: str, summary: str, description: str,
#                       assignees: List[str], link_blocker: Optional[str] = None,
#                       simulateFailure: bool = False) -> dict:
#     m, u = _resolve_op("createTicket")
#     payload = {"project": project, "summary": summary, "description": description,
#                "assignees": assignees, "link_blocker": link_blocker,
#                "simulateFailure": simulateFailure}
#     return requests.request(m, u, json=payload, timeout=10).json()

# def jira_getTicketLogs(transactionId: str) -> dict:
#     m, u = _resolve_op("getTicketLogs")
#     return requests.request(m, u, json={"transactionId": transactionId}, timeout=10).json()

# root_agent = Agent(
#     name="orchestrator_agent",
#     model="gemini-2.0-flash-001",
#     instruction=("Réponds en français. Utilise jira_findRisks(project), "
#                  "jira_createTicket(...), jira_getTicketLogs(transactionId)."),
#     tools=[jira_findRisks, jira_createTicket, jira_getTicketLogs],
# )

# --- remplace ton resolver et tes wrappers par ceci ---
import requests
from typing import List, Optional,Set, Dict, Any
from google.adk.agents import Agent


BASE = "http://127.0.0.1:9002"  # adk api_server jira_fake_agent
_spec_cache = None

def _load_spec() -> Dict[str, Any]:
    global _spec_cache
    if _spec_cache is None:
        _spec_cache = requests.get(f"{BASE}/openapi.json", timeout=5).json()
    return _spec_cache

def _iter_ops():
    """Itère (METHOD, FULL_URL, operationObject, pathTemplate)."""
    spec = _load_spec()
    for path, methods in (spec.get("paths") or {}).items():
        for method, op in (methods or {}).items():
            yield (method.upper(), f"{BASE}{path}", op or {}, path)

def _schema_keys(schema: Dict[str, Any]) -> Set[str]:
    """Collecte récursivement TOUTES les clés présentes dans un schema JSON."""
    keys: Set[str] = set()

    def walk(s: Any):
        if not isinstance(s, dict):
            return
        props = s.get("properties") or {}
        for k, v in props.items():
            keys.add(k)
            walk(v)
        if "items" in s:
            walk(s["items"])
        for alt in ("anyOf", "oneOf", "allOf"):
            if alt in s and isinstance(s[alt], list):
                for ss in s[alt]:
                    walk(ss)

    walk(schema or {})
    return keys

def _props_for_op(op: Dict[str, Any]) -> Set[str]:
    """Retourne l'ensemble des noms de propriétés attendues (body + query/path)."""
    keys: Set[str] = set()
    # requestBody (tous media types)
    content = (op.get("requestBody") or {}).get("content") or {}
    for _, media_obj in content.items():
        schema = (media_obj or {}).get("schema") or {}
        keys |= _schema_keys(schema)
    # paramètres query/path
    for p in op.get("parameters", []) or []:
        if p.get("in") in ("query", "path") and "name" in p:
            keys.add(p["name"])
        keys |= _schema_keys(p.get("schema") or {})
    return keys

def _resolve_by_keys(required_any: Set[str], nice_to_have: Set[str] = set()):
    """
    Choisit l’opération dont le schéma contient au moins les clés de `required_any`
    (où qu’elles soient : body, nested, query). Scoring :
      +10 par clé required présente, +1 par clé nice-to-have
      +50 si TOUTES les required sont présentes
      +2 si méthode == POST
    Retourne: (HTTP_METHOD, FULL_URL)
    """
    candidates = []  # (score, method, url, path, found_keys)

    for method, url, op, path in _iter_ops():
        props = _props_for_op(op)
        if not props:
            continue

        score = 0
        present_required = {k for k in required_any if k in props}
        present_nice = {k for k in nice_to_have if k in props}

        score += 10 * len(present_required)
        score += 1 * len(present_nice)
        if required_any and required_any.issubset(props):
            score += 50
        if method == "POST":
            score += 2

        if score > 0:
            candidates.append((score, method, url, path, props))

    if not candidates:
        # aide au debug: liste rapidement ce qu'on voit
        spec = _load_spec()
        # Décommente si tu veux regarder côté console:
        # for m,u,op,p in _iter_ops():
        #     print(f"{m} {p} -> keys={sorted(list(_props_for_op(op)))}")

        raise RuntimeError("Aucune opération OpenAPI ne correspond aux clés attendues.")

    candidates.sort(key=lambda x: (-x[0], x[3]))
    _, m, full, _, _ = candidates[0]
    return m, full


# ---- Wrappers HTTP (A2A réseau) basés sur la détection par schéma ----

def jira_findRisks(project: str) -> dict:
    # findRisks attend au moins: {"project"}
    m, u = _resolve_by_keys(required_any={"project"})
    resp = requests.request(m, u, json={"project": project}, timeout=10)
    return resp.json()

def jira_createTicket(
    project: str,
    summary: str,
    description: str,
    assignees: List[str],
    link_blocker: Optional[str] = None,
    simulateFailure: bool = False,
) -> dict:
    # createTicket ressemble à: {"project","summary","description","assignees", "link_blocker?","simulateFailure?"}
    m, u = _resolve_by_keys(
        required_any={"project", "summary", "description", "assignees"},
        nice_to_have={"link_blocker", "simulateFailure"},
    )
    payload = {
        "project": project,
        "summary": summary,
        "description": description,
        "assignees": assignees,
        "link_blocker": link_blocker,
        "simulateFailure": simulateFailure,
    }
    resp = requests.request(m, u, json=payload, timeout=10)
    return resp.json()

def jira_getTicketLogs(transactionId: str) -> dict:
    # getTicketLogs attend: {"transactionId"}
    m, u = _resolve_by_keys(required_any={"transactionId"})
    resp = requests.request(m, u, json={"transactionId": transactionId}, timeout=10)
    return resp.json()


root_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.0-flash-001",
    instruction=("Réponds en français. Utilise jira_findRisks(project), "
                 "jira_createTicket(...), jira_getTicketLogs(transactionId)."),
    tools=[jira_findRisks, jira_createTicket, jira_getTicketLogs],
)