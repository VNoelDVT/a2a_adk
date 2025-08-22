# sk_orchestrator/sk_gateway.py
import os
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.kernel import KernelArguments

# SK function-calling (multi-version)
try:
    from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
    HAVE_FCB = True
except Exception:
    HAVE_FCB = False

# Exposition des fonctions au LLM
from semantic_kernel.functions import kernel_function

# Nos fonctions A2A (lisent l’Agent Card Jira, appellent l’OpenAPI du fake Jira)
from jira_a2a_plugin import find_risks as _find_risks
from jira_a2a_plugin import create_ticket as _create_ticket
from jira_a2a_plugin import get_logs as _get_logs


class JiraPlugin:
    """Plugin SK exposant les 3 tools Jira via @kernel_function."""

    @kernel_function(
        name="findRisks",
        description="Trouver les risques d'un projet (stalled/high/blockers/controversial).",
    )
    def find_risks(self, project: str, stalled_days: int = 5) -> Dict[str, Any]:
        return _find_risks(project, stalled_days)

    @kernel_function(
        name="createTicket",
        description="Créer un ticket (assignees, lien 'blocker', échec simulé).",
    )
    def create_ticket(
        self,
        project: str,
        summary: str,
        description: str,
        assignees: List[str],
        link_blocker: Optional[str] = None,
        simulateFailure: bool = False,
    ) -> Dict[str, Any]:
        return _create_ticket(project, summary, description, assignees, link_blocker, simulateFailure)

    @kernel_function(
        name="getTicketLogs",
        description="Récupérer les logs d'un job par transactionId.",
    )
    def get_ticket_logs(self, transactionId: str) -> Dict[str, Any]:
        return _get_logs(transactionId)


async def build_kernel() -> Kernel:
    """Construit le Kernel SK branché sur NVIDIA + plugin Jira."""
    load_dotenv()

    base_url = os.getenv("NVIDIA_BASE_URL") or os.getenv("base_url") or "https://integrate.api.nvidia.com/v1"
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY manquant. Ajoute-le dans .env.")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    kernel = Kernel()
    svc = OpenAIChatCompletion(ai_model_id=model, async_client=client, service_id="nvidia")
    kernel.add_service(svc)

    kernel.add_plugin(JiraPlugin(), plugin_name="jira")
    return kernel


# ------------------- FastAPI -------------------
app = FastAPI(title="SK Orchestrator Gateway", version="1.0.0")


class ChatIn(BaseModel):
    input: str


class ChatOut(BaseModel):
    output: str


@app.on_event("startup")
async def _startup():
    app.state.kernel = await build_kernel()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatOut, summary="Chat orchestré (function-calling auto)")
async def chat(body: ChatIn):
    kernel: Kernel = app.state.kernel
    if HAVE_FCB:
        args = KernelArguments(
            service_id="nvidia",
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )
        result = await kernel.invoke_prompt(body.input, arguments=args)
    else:
        result = await kernel.invoke_prompt(body.input)
    return ChatOut(output=str(result))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9100)
