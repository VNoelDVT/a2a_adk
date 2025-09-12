# orchestrator/nvidia_runner.py
import os, json
from typing import Any, Dict, List
from openai import OpenAI

# --- import your existing tools ---------------------------------------------
from agents.intake_normalizer_agent import detect_and_normalize
from agents.offer_vectorizer_agent import make_offer_vector
from agents.candidate_matcher_agent import match_candidates
from agents.capacity_checker_agent import check_capacity_for_list
from agents.geo_filter_agent import filter_by_geo
from agents.finance_policy_agent import enrich_finance_policy
from agents.explainability_agent import explain_shortlist  # optional
from agents.bu_health_agent import bu_overview
from orchestrator.ui_contracts import shortlist_view, bu_health_view

# --- OpenAI (NVIDIA) client --------------------------------------------------
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY  = os.environ["OPENAI_API_KEY"]
MODEL    = os.getenv("OPENAI_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Tool registry: map tool names <-> python callables ----------------------
# Keep names short and stable; they are what the model will call.
TOOL_IMPLS = {
    "detect_and_normalize": detect_and_normalize,
    "make_offer_vector":     make_offer_vector,
    "match_candidates":      match_candidates,
    "check_capacity_for_list": check_capacity_for_list,
    "filter_by_geo":         filter_by_geo,
    "enrich_finance_policy": enrich_finance_policy,
    "explain_shortlist":     explain_shortlist,  # optional step
    "bu_overview":           bu_overview,
    "shortlist_view":        shortlist_view,     # FINAL
    "bu_health_view":        bu_health_view,     # FINAL
}

# --- OpenAI tool schemas (JSON Schema v7) ------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_and_normalize",
            "description": "Parse user input to a normalized offer or BU intent.",
            "parameters": {
                "type": "object",
                "properties": { "input": {"type": "string"} },
                "required": ["input"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_offer_vector",
            "description": "Build weighted vector structure from offer.",
            "parameters": {
                "type": "object",
                "properties": { "offer": {"type": "object"} },
                "required": ["offer"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_candidates",
            "description": "Given a vector (or wrapper with vector), return top candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vector": {"type": "object"},
                    "top_n":  {"type": "integer"}
                },
                "required": ["vector"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_capacity_for_list",
            "description": "Filter/annotate items by capacity vs start date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {"type": "array"},
                    "start_by": {"type": "string"}
                },
                "required": ["items", "start_by"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_geo",
            "description": "Filter shortlist by max distance from location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {
                            "shortlist": {"type": "array"},
                            "location":  {"type": "string"},
                            "max_km":    {"type": "number"}
                        },
                        "required": ["shortlist", "location"],
                        "additionalProperties": True
                    }
                },
                "required": ["payload"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enrich_finance_policy",
            "description": "Apply finance/policy constraints to shortlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": {"type": "object"},
                    "shortlist": {"type": "array"}
                },
                "required": ["offer", "shortlist"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_shortlist",
            "description": "(Optional) Generate a brief explanation of match quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": {"type": "object"},
                    "shortlist": {"type": "array"}
                },
                "required": ["offer", "shortlist"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bu_overview",
            "description": "Fetch/aggregate BU KPIs into a report dict.",
            "parameters": {
                "type": "object",
                "properties": { "bu_name": {"type": "string"} },
                "required": ["bu_name"],
                "additionalProperties": False
            },
        },
    },
    # FINAL views (these return the final text to show to the user)
    {
        "type": "function",
        "function": {
            "name": "shortlist_view",
            "description": "Render final shortlist UI as plain text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": {"type": "object"},
                    "shortlist": {"type": "array"}
                },
                "required": ["offer", "shortlist"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bu_health_view",
            "description": "Render final BU health report as plain text.",
            "parameters": {
                "type": "object",
                "properties": { "report": {"type": "object"} },
                "required": ["report"],
                "additionalProperties": True
            },
        },
    },
]

# --- Tool dispatcher ----------------------------------------------------------
def _call_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = TOOL_IMPLS[name]
    # Allow slight arg name mismatch the model might produce
    if name == "filter_by_geo" and "payload" in args:
        return fn(args["payload"])
    if name == "match_candidates":
        # our function accepts (vector, top_n=..)
        return fn(args.get("vector") or args, args.get("top_n", 5))
    return fn(**args)

# --- Orchestrated run with NVIDIA --------------------------------------------
SYSTEM_PROMPT = (
    "Tu es l’orchestrateur. Tu n’écris pas de texte libre tant que la vue finale n’est pas prête. "
    "Tu enchaînes EXCLUSIVEMENT des appels d’outils selon les cas suivants.\n\n"
    "Cas OFFRE -> CANDIDATS:\n"
    "  1) detect_and_normalize(input)\n"
    "  2) make_offer_vector(offer)\n"
    "  3) match_candidates(vector)\n"
    "  4) check_capacity_for_list(items=candidates, start_by=offer.start_by)\n"
    "  5) filter_by_geo(payload={shortlist, location=offer.location, max_km=400})\n"
    "  6) enrich_finance_policy(offer, shortlist)\n"
    "  7) (optionnel) explain_shortlist(offer, shortlist)\n"
    "  8) shortlist_view(offer, shortlist)  # puis STOP\n\n"
    "Cas BU:\n"
    "  1) detect_and_normalize(input)\n"
    "  2) bu_overview(bu_name)\n"
    "  3) bu_health_view(report)  # puis STOP\n"
)

def run_with_nvidia(user_text: str, max_steps: int = 16) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]

    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.6,
            top_p=0.7,
        )
        msg = resp.choices[0].message

        # If the model produced a direct text answer (should only be final), return it.
        if msg.content and not msg.tool_calls:
            return msg.content

        # Execute tool calls (support multiple per turn)
        tool_results_msgs: List[Dict[str, Any]] = []
        for tool_call in msg.tool_calls or []:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            result = _call_tool(name, args)

            # FINAL views: return plain string immediately
            if name in ("shortlist_view", "bu_health_view"):
                if isinstance(result, dict) and "text" in result:
                    return str(result["text"])
                return str(result)

            # Otherwise, append tool result for the next LLM step
            tool_results_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
        messages.extend(tool_results_msgs)

    # Safety fallback
    return "Processus interrompu (trop d'étapes)."

# Convenience alias
run = run_with_nvidia
