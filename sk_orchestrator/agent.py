# # sk_orchestrator/agent.py
# import os
# import httpx

# from google.adk.agents import Agent
# from google.adk.tools import FunctionTool

# # ---- Try a few ADK variants for the Gemini LLM class ----
# LLMCls = None
# LLMVariant = None
# _import_errors = []

# for attempt in (
#     ("google.adk.models.google_llm", "GoogleLLM", "GoogleLLMVariant"),
#     ("google.adk.models.gemini_llm", "GeminiLLM", "GeminiLLMVariant"),
#     ("google.adk.models.google_genai_llm", "GoogleGenAILLM", "GoogleGenAILLMVariant"),
# ):
#     mod_path, cls_name, var_name = attempt
#     try:
#         mod = __import__(mod_path, fromlist=[cls_name, var_name])
#         LLMCls = getattr(mod, cls_name)
#         LLMVariant = getattr(mod, var_name)
#         break
#     except Exception as e:
#         _import_errors.append(f"{mod_path}.{cls_name}: {e}")

# if LLMCls is None:
#     raise ImportError(
#         "Unable to import a Google/Gemini LLM class from ADK. Tried:\n  - "
#         + "\n  - ".join(_import_errors)
#     )

# # -----------------------------------------------------------------

# GATEWAY_URL = os.getenv("SK_GATEWAY_URL", "http://127.0.0.1:9100/chat")

# def sk_chat(input: str) -> dict:
#     """
#     Send the user prompt to the SK gateway (NVIDIA backend) and return its reply.
#     """
#     try:
#         with httpx.Client(timeout=30.0) as client:
#             r = client.post(GATEWAY_URL, json={"input": input})
#             r.raise_for_status()
#             data = r.json()
#             return {"status": "success", "output": data.get("output", "")}
#     except Exception as e:
#         return {"status": "error", "error_message": str(e)}

# sk_chat_tool = FunctionTool(
#     name="sk_chat",
#     description="Send a prompt to the Semantic Kernel gateway and get the answer.",
#     function=sk_chat,   # some ADK versions use `function=`; this one is compatible
# )

# root_agent = Agent(
#     model=LLMCls(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001"),
#                  variant=getattr(LLMVariant, "GEMINI_API", LLMVariant.GEMINI_API)),
#     tools=[sk_chat_tool],
#     system_instruction=(
#         "You are a proxy. For each user message, call the tool 'sk_chat' exactly once "
#         "with the full message in the 'input' argument. If the tool returns status=success, "
#         "return ONLY the 'output' field. If it returns status=error, briefly explain the error."
#     ),
# )


# orchestrator_agent/agent.py
import os
import httpx

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.models.google_llm import Gemini as GoogleLLM
from google.adk.utils.variant_utils import GoogleLLMVariant

SK_GATEWAY_URL = os.getenv("SK_GATEWAY_URL", "http://127.0.0.1:9100/chat")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
JIRA_BASE = os.getenv("JIRA_BASE", "http://127.0.0.1:9002")

def sk_chat(input: str) -> dict:
    """
    Send a prompt to the Semantic Kernel gateway (NVIDIA backend) and get the answer.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(SK_GATEWAY_URL, json={"input": input})
            r.raise_for_status()
            data = r.json()
            return {"status": "success", "output": data.get("output", "")}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def jira_find_risks(project_id: str) -> dict:
    """
    Find risky tasks for a project (e.g., 'Phoenix-V2' or 'Phoenix_V2').
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{JIRA_BASE}/findRisks", params={"project_id": project_id})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"{e.response.status_code} {e.response.text}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

def jira_create_ticket(
    project_id: str,
    title: str,
    description: str = "",
    assignees_csv: str = "",
    link_blocker: str = "",
    simulate_failure: bool = False,
) -> dict:
    """
    Create a (fake) JIRA ticket via Agent B.
    - assignees_csv: comma-separated assignees
    - link_blocker: ticket key to link as blocker (empty string if none)
    - simulate_failure: set True to simulate a 403 failure
    """
    assignees = [a.strip() for a in assignees_csv.split(",")] if assignees_csv else []
    link_blocker_value = link_blocker.strip() or None

    payload = {
        "project": project_id,
        "summary": title,
        "description": description,
        "assignees": assignees,
        "link_blocker": link_blocker_value,
        "simulateFailure": bool(simulate_failure),
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(f"{JIRA_BASE}/createTicket", json=payload)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"{e.response.status_code} {e.response.text}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


def jira_get_ticket_logs(transaction_id: str) -> dict:
    """
    Get logs for a given transaction_id from Agent B.
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{JIRA_BASE}/getTicketLogs", params={"transaction_id": transaction_id})
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"{e.response.status_code} {e.response.text}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


sk_chat_tool = FunctionTool(sk_chat)
jira_find_risks_tool = FunctionTool(jira_find_risks)
jira_create_ticket_tool = FunctionTool(jira_create_ticket)
jira_get_ticket_logs_tool = FunctionTool(jira_get_ticket_logs)

root_agent = Agent(
    name="sk_orchestrator",
    model=GoogleLLM(
        model=GEMINI_MODEL,
        variant=GoogleLLMVariant.GEMINI_API,
    ),
    tools=[
        jira_find_risks_tool,
        jira_create_ticket_tool,
        jira_get_ticket_logs_tool,
        sk_chat_tool,
    ],
    instruction=(
        "You orchestrate work across a JIRA-like Agent.\n"
        "- For risk analysis, call 'jira_find_risks(project_id)'.\n"
        "- To open an issue, call 'jira_create_ticket(project_id, title, description, assignees_csv, link_blocker, simulateFailure)'.\n"
        "- To audit/replay, call 'jira_get_ticket_logs(transaction_id)'.\n"
        "Always summarize what you did and include any transactionId returned by tools."
    ),
)

