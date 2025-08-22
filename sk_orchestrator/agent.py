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
from google.adk.models.google_llm import GoogleLLM, GoogleLLMVariant  # this worked earlier for you

SK_GATEWAY_URL = os.getenv("SK_GATEWAY_URL", "http://127.0.0.1:9100/chat")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

def sk_chat(input: str) -> dict:
    """
    Forwards a user prompt to the local Semantic Kernel gateway (which uses NVIDIA NIM)
    and returns its answer.
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(SK_GATEWAY_URL, json={"input": input})
            r.raise_for_status()
            data = r.json()
            return {"status": "success", "output": data.get("output", "")}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

sk_chat_tool = FunctionTool(
    name="sk_chat",
    description="Send a prompt to the SK gateway (NVIDIA backend) and get the answer.",
    function=sk_chat,
)

root_agent = Agent(
    model=GoogleLLM(
        model=GEMINI_MODEL,
        variant=GoogleLLMVariant.GEMINI_API,
    ),
    tools=[
        # keep your existing Jira tools here
        sk_chat_tool,
    ],
    system_instruction=(
        "You are an orchestrator. When the user asks anything that should be handled "
        "by the NVIDIA model, call the tool 'sk_chat' exactly once with the full user "
        "message in the 'input' argument. If the tool returns status=success, answer "
        "with the 'output'. If it returns status=error, explain briefly."
    ),
)
