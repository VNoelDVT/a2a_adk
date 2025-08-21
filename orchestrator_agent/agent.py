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

from typing import List, Optional

from google.adk.agents import Agent

# --- Essai n°1 : via OpenAPI (A2A réseau) ---
JIRA_FAKE_OPENAPI = "http://127.0.0.1:9002/openapi.json"
_tools = None
_reason = None

try:
    # Selon les versions d'ADK, le chemin du tool change :
    try:
        from google.adk.tools.openapi_tool import OpenAPITool  # ADK récent
    except Exception:
        from google.adk.tools.open_api_tool import OpenAPITool  # ADK plus ancien

    jira_tool = OpenAPITool.from_url(
        JIRA_FAKE_OPENAPI,
        name="jira_fake",
        description="Outils de l'agent Jira factice: findRisks, createTicket, getTicketLogs."
    )
    _tools = [jira_tool]

except Exception as e:
    # --- Fallback : in-process (A2A léger) ---
    _reason = f"OpenAPITool indisponible: {e}. Fallback in-process."
    from jira_fake_agent.agent import findRisks, createTicket, getTicketLogs

    def jira_findRisks(project: str) -> dict:
        """Analyse les risques d'un projet (stalled, high priority, blockers, controversial)."""
        return findRisks(project)

    def jira_createTicket(
        project: str,
        summary: str,
        description: str,
        assignees: list[str],
        link_blocker: Optional[str] = None,
        simulateFailure: bool = False,
    ) -> dict:
        return createTicket(project, summary, description, assignees, link_blocker, simulateFailure)

    def jira_getTicketLogs(transactionId: str) -> dict:
        """Retourne les logs détaillés d'un job par transactionId."""
        return getTicketLogs(transactionId)

    _tools = [jira_findRisks, jira_createTicket, jira_getTicketLogs]

# --- Agent orchestrateur ---
root_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.0-flash-001",
    description="Orchestrateur A2A pour gestion de risques projet.",
    instruction=(
        "Réponds en français. "
        "Si disponible, utilise le tool OpenAPI 'jira_fake' (findRisks, createTicket, getTicketLogs). "
        "Sinon, utilise les outils locaux du fallback aux mêmes noms et signatures. "
        "Pour les actions longues ou ayant échoué, renvoie le transactionId et explique comment obtenir les logs."
        + (f"  NOTE TECHNIQUE: {_reason}" if _reason else "")
    ),
    tools=_tools,
)

