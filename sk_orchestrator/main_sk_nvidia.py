# import os, asyncio
# from openai import AsyncOpenAI
# from semantic_kernel import Kernel
# from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
# from semantic_kernel.functions import KernelFunction
# from semantic_kernel.connectors.ai.function_calling import FunctionChoiceBehavior
# from semantic_kernel.kernel import KernelArguments

# from jira_a2a_plugin import find_risks, create_ticket, get_logs

# import os

# NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL") or os.getenv("base_url") or "https://integrate.api.nvidia.com/v1"
# NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
# NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")

# from dotenv import load_dotenv
# load_dotenv()


# async def build_kernel() -> Kernel:
#     # Client OpenAI-compatible pointé vers NVIDIA
#     client = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

#     kernel = Kernel()
#     # Service LLM NVIDIA pour SK
#     svc = OpenAIChatCompletion(
#         ai_model_id=NVIDIA_MODEL,
#         async_client=client,
#         service_id="nvidia"  # identifiant interne SK
#     )
#     kernel.add_service(svc)

#     # Enregistrer nos 3 fonctions (tools) exposées au LLM
#     kernel.add_function(
#         plugin_name="jira",
#         function=KernelFunction.from_native_function(
#             find_risks,
#             name="findRisks",
#             description="Trouver les risques d'un projet Jira (stalled/high/blockers/controversial).",
#             parameters=[
#                 {"name":"project", "type":"string", "description":"ID projet (ex: Phoenix_V2)"},
#                 {"name":"stalled_days", "type":"integer", "description":"Seuil de jours pour 'stalled'", "required":False},
#             ],
#         ),
#     )
#     kernel.add_function(
#         plugin_name="jira",
#         function=KernelFunction.from_native_function(
#             create_ticket,
#             name="createTicket",
#             description="Créer un ticket (priorité, assignees, lien 'blocker', échec simulé éventuel).",
#             parameters=[
#                 {"name":"project", "type":"string"},
#                 {"name":"summary", "type":"string"},
#                 {"name":"description", "type":"string"},
#                 {"name":"assignees", "type":"array"},
#                 {"name":"link_blocker", "type":"string", "required":False},
#                 {"name":"simulateFailure", "type":"boolean", "required":False},
#             ],
#         ),
#     )
#     kernel.add_function(
#         plugin_name="jira",
#         function=KernelFunction.from_native_function(
#             get_logs,
#             name="getTicketLogs",
#             description="Récupérer les logs d'un job par transactionId.",
#             parameters=[{"name":"transactionId","type":"string"}],
#         ),
#     )
#     return kernel

# async def chat_loop():
#     kernel = await build_kernel()
#     print("Orchestrateur SK+NVIDIA prêt. Tape 'exit' pour quitter.\n")
#     while True:
#         user = input("> ")
#         if not user or user.lower() in {"exit","quit"}:
#             break

#         # LLM NVIDIA + tool-calling AUTO (les 3 fonctions 'jira.*' sont exposées)
#         args = KernelArguments(
#             service_id="nvidia",
#             function_choice_behavior=FunctionChoiceBehavior.Auto()
#         )
#         result = await kernel.invoke_prompt(user, arguments=args)
#         print(str(result), "\n")

# if __name__ == "__main__":
#     asyncio.run(chat_loop())

import os, asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.kernel import KernelArguments

# Import multi-version : FunctionChoiceBehavior (sinon on s'en passe)
try:
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    HAVE_FCB = True
except Exception:
    HAVE_FCB = False

# Le décorateur pour exposer des fonctions au LLM
try:
    from semantic_kernel.functions import kernel_function
except Exception as e:
    raise RuntimeError(
        "Ta version de Semantic Kernel est trop ancienne (pas de 'kernel_function'). "
        "Mets à jour: pip install -U semantic-kernel"
    ) from e

# Charge .env
load_dotenv()

# NVIDIA (OpenAI-compatible)
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL") or os.getenv("base_url") or "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY manquant (mets-le dans .env ou tes variables d'env).")

# Notre plugin A2A -> Jira (fonctions déjà prêtes)
from jira_a2a_plugin import find_risks as _find_risks
from jira_a2a_plugin import create_ticket as _create_ticket
from jira_a2a_plugin import get_logs as _get_logs

class JiraPlugin:
    """Expose les fonctions au LLM via @kernel_function."""

    @kernel_function(
        name="findRisks",
        description="Trouver les risques d'un projet (stalled/high/blockers/controversial)."
    )
    def find_risks(self, project: str, stalled_days: int = 5) -> dict:
        return _find_risks(project, stalled_days)

    @kernel_function(
        name="createTicket",
        description="Créer un ticket (assignees, lien 'blocker', échec simulé)."
    )
    def create_ticket(
        self,
        project: str,
        summary: str,
        description: str,
        assignees: list,
        link_blocker: str | None = None,
        simulateFailure: bool = False,
    ) -> dict:
        return _create_ticket(project, summary, description, assignees, link_blocker, simulateFailure)

    @kernel_function(
        name="getTicketLogs",
        description="Récupérer les logs d'un job par transactionId."
    )
    def get_ticket_logs(self, transactionId: str) -> dict:
        return _get_logs(transactionId)

async def build_kernel() -> Kernel:
    # Client OpenAI-compatible pointé vers NVIDIA
    client = AsyncOpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

    kernel = Kernel()
    svc = OpenAIChatCompletion(ai_model_id=NVIDIA_MODEL, async_client=client, service_id="nvidia")
    kernel.add_service(svc)

    # Enregistrer le plugin "jira" (les 3 méthodes deviennent des tools)
    kernel.add_plugin(JiraPlugin(), plugin_name="jira")
    return kernel

async def chat_loop():
    kernel = await build_kernel()
    print("Orchestrateur SK + NVIDIA prêt. Tape 'exit' pour quitter.\n")
    while True:
        user = input("> ")
        if not user or user.lower() in {"exit", "quit"}:
            break

        # Tool calling automatique si dispo, sinon appel simple
        if HAVE_FCB:
            args = KernelArguments(
                service_id="nvidia",
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
            result = await kernel.invoke_prompt(user, arguments=args)
        else:
            # Fallback (certaines versions déclenchent quand même le tool-calling)
            result = await kernel.invoke_prompt(user)

        print(str(result), "\n")

if __name__ == "__main__":
    asyncio.run(chat_loop())
