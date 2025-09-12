# agents/nvidia_orchestrator_tool.py
import os, json, inspect, time, random
from typing import Any, Dict
from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError
from google.adk.tools import ToolContext

# --- import your existing python tools ---
from agents.intake_normalizer_agent import detect_and_normalize
from agents.offer_vectorizer_agent import make_offer_vector
from agents.candidate_matcher_agent import match_candidates
from agents.capacity_checker_agent import check_capacity_for_list
from agents.geo_filter_agent import filter_by_geo
from agents.finance_policy_agent import enrich_finance_policy
from agents.explainability_agent import explain_shortlist
from agents.bu_health_agent import bu_overview
from orchestrator.ui_contracts import shortlist_view, bu_health_view

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY  = os.environ["OPENAI_API_KEY"]
MODEL    = os.getenv("OPENAI_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
_client  = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- tool declarations (JSON Schema) ---
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "detect_and_normalize",
            "description": "Parse user text and return a normalized intent (offer query or BU query).",
            "parameters": {
                "type": "object",
                "properties": { "input": { "type": "string" } },
                "required": ["input"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_offer_vector",
            "description": "Build a weighted vector from a normalized OfferSpec.",
            "parameters": {
                "type": "object",
                "properties": { "offer": { "type": "object" } },
                "required": ["offer"],
                "additionalProperties": True
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_candidates",
            "description": "Given an offer vector (or wrapper containing it), return top candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vector": { "type": "object" },
                    "top_n":  { "type": "integer" }
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
            "description": "Check capacity of items relative to a start date and filter/annotate the list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": { "type": "array" },
                    "start_by": { "type": "string" }
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
            "description": "Filter shortlist by distance from a location (in km) and return the filtered payload.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {
                            "shortlist": { "type": "array" },
                            "location":  { "type": "string" },
                            "max_km":    { "type": "number" }
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
            "description": "Apply finance/policy rules to an offer + shortlist and return enriched data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": { "type": "object" },
                    "shortlist": { "type": "array" }
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
            "description": "(Optional) Produce a short textual explanation for why the shortlist fits the offer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": { "type": "object" },
                    "shortlist": { "type": "array" }
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
            "description": "Fetch and aggregate KPIs for a given BU name and return a report dict.",
            "parameters": {
                "type": "object",
                "properties": { "bu_name": { "type": "string" } },
                "required": ["bu_name"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shortlist_view",
            "description": "Render the final shortlist as a plain text string for display.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer": { "type": "object" },
                    "shortlist": { "type": "array" }
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
            "description": "Render the final BU health report as a plain text string for display.",
            "parameters": {
                "type": "object",
                "properties": { "report": { "type": "object" } },
                "required": ["report"],
                "additionalProperties": True
            },
        },
    },
]

_IMPLS = {
    "detect_and_normalize": detect_and_normalize,
    "make_offer_vector": make_offer_vector,
    "match_candidates": lambda vector, top_n=5: match_candidates(vector, top_n),
    "check_capacity_for_list": check_capacity_for_list,
    "filter_by_geo": lambda payload: filter_by_geo(payload),
    "enrich_finance_policy": enrich_finance_policy,
    "explain_shortlist": explain_shortlist,
    "bu_overview": bu_overview,
    "shortlist_view": shortlist_view,
    "bu_health_view": bu_health_view,
}

_SYSTEM = (
    "Tu es l’orchestrateur. Enchaîne UNIQUEMENT des appels d’outils.\n"
    "OFFRE→CANDIDATS: 1) detect_and_normalize(input) 2) make_offer_vector(offer) "
    "3) match_candidates(vector) 4) check_capacity_for_list(items, start_by) "
    "5) filter_by_geo(payload={shortlist, location, max_km=400}) "
    "6) enrich_finance_policy(offer, shortlist) 7) (opt) explain_shortlist "
    "8) shortlist_view(offer, shortlist) puis STOP.\n"
    "BU: 1) detect_and_normalize(input) 2) bu_overview(bu_name) 3) bu_health_view(report) puis STOP."
)

def _maybe_with_tool_context(fn, kwargs: Dict[str, Any]) -> Any:
    """If fn accepts 'tool_context' and it's not provided, pass None."""
    try:
        sig = inspect.signature(fn)
        if "tool_context" in sig.parameters and "tool_context" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["tool_context"] = None
    except Exception:
        pass
    return fn(**kwargs)

def _call_tool(name: str, args: Dict[str, Any]) -> Any:
    fn = _IMPLS[name]

    # --- Robust mapping for detect_and_normalize -----------------------------
    if name == "detect_and_normalize":
        txt = (
            args.get("input")
            or args.get("text")
            or args.get("query")
            or args.get("message")
            or args.get("user_input")
            or (next(iter(args.values())) if args else None)
        )
        if txt is None:
            raise ValueError("detect_and_normalize: missing input text")
        try:
            return fn(txt)  # positional
        except TypeError:
            for kw in ("text", "query", "user_text", "utterance"):
                try:
                    return _maybe_with_tool_context(fn, {kw: txt})
                except TypeError:
                    pass
            sig = inspect.signature(fn)
            pname = next((p.name for p in sig.parameters.values() if p.name != "tool_context"), "text")
            return _maybe_with_tool_context(fn, {pname: txt})

    # --- match_candidates: accept wrapper or raw vector ----------------------
    if name == "match_candidates":
        vec = args.get("vector") or args
        top_n = args.get("top_n", 5)
        return fn(vec, top_n)

    # --- filter_by_geo: payload or split args --------------------------------
    if name == "filter_by_geo":
        payload = args.get("payload") or args
        try:
            return _maybe_with_tool_context(fn, {"payload": payload})
        except TypeError:
            return _maybe_with_tool_context(fn, {
                "shortlist": payload["shortlist"],
                "location":  payload.get("location"),
                "max_km":    payload.get("max_km", 400),
            })

    # --- Final views: allow tool_context; return plain string ----------------
    if name in ("shortlist_view", "bu_health_view"):
        out = _maybe_with_tool_context(fn, args)
        return out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else str(out))

    # --- Default path ---------------------------------------------------------
    try:
        return fn(**args)
    except TypeError:
        return _maybe_with_tool_context(fn, args)

# --- resilient NVIDIA call (retries on 5xx/429) ------------------------------
def _nv_chat(messages, tools, max_retries: int = 4):
    for attempt in range(max_retries + 1):
        try:
            return _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                top_p=0.9,
                max_tokens=2048,
            )
        except (RateLimitError, InternalServerError, APIError) as e:
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503, 504) or isinstance(e, (RateLimitError, InternalServerError)):
                if attempt < max_retries:
                    time.sleep((0.6 * (2 ** attempt)) + random.uniform(0, 0.3))
                    continue
            raise

def nvidia_orchestrate(input: str, tool_context: ToolContext = None) -> str:
    """
    Single ADK tool that runs the whole orchestration on NVIDIA and returns final text.
    ADK will print this string as a chat bubble (summarization ENABLED).
    """
    # ✅ Let ADK print the tool return as the final assistant message.
    try:
        if tool_context is not None:
            tool_context.actions.skip_summarization = False
    except Exception:
        pass

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": input},
    ]

    try:
        for _ in range(20):
            resp = _nv_chat(messages, _TOOLS)
            msg = resp.choices[0].message

            # If the model produced direct text (end condition), return it.
            if msg.content and not msg.tool_calls:
                return msg.content

            tool_msgs = []
            for tc in msg.tool_calls or []:
                args = json.loads(tc.function.arguments or "{}")
                out = _call_tool(tc.function.name, args)

                # FINAL views → return immediately as plain string
                if tc.function.name in ("shortlist_view", "bu_health_view"):
                    return out if isinstance(out, str) else (out.get("text") if isinstance(out, dict) else str(out))

                tool_msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(out, ensure_ascii=False)
                })

            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            messages.extend(tool_msgs)

        return "Processus interrompu (trop d'étapes)."

    except (RateLimitError, InternalServerError, APIError) as e:
        code = getattr(e, "status_code", "N/A")
        return f"(NVIDIA endpoint a renvoyé une erreur {code}. Réessayez dans un instant.)"
    except Exception as e:
        return f"(Erreur inattendue côté NVIDIA bridge: {type(e).__name__}: {e})"
