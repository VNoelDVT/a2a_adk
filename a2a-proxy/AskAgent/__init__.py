# AskAgent/__init__.py — AAD (Managed Identity) + Agents v1 + robust text extraction
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".python-packages", "lib", "site-packages"))

import os, json, time, logging, traceback, requests
import azure.functions as func


API_VERSION = "v1"

def _env(k): 
    return (os.getenv(k) or "").strip()

PROJECT_ENDPOINT = _env("PROJECT_ENDPOINT").rstrip("/")   # e.g. https://...services.ai.azure.com/api/projects/YourProject
AGENT_ID         = _env("AGENT_ID")                       # e.g. asst_...

def _http(code, obj):
    return func.HttpResponse(json.dumps(obj), status_code=code, mimetype="application/json")

# --- Managed Identity (AAD) token for https://ai.azure.com ---
def _get_msi_token(resource="https://ai.azure.com"):
    ident_ep  = os.getenv("IDENTITY_ENDPOINT")
    ident_hdr = os.getenv("IDENTITY_HEADER")
    if ident_ep and ident_hdr:
        try:
            r = requests.get(
                ident_ep,
                params={"resource": resource, "api-version": "2019-08-01"},
                headers={"X-IDENTITY-HEADER": ident_hdr},
                timeout=1.5,
            )
            if r.ok:
                return r.json().get("access_token")
            logging.warning("MSI (IDENTITY_ENDPOINT) %s %s", r.status_code, r.text[:200])
        except Exception:
            logging.exception("MSI endpoint exception")
    # IMDS fallback
    try:
        r = requests.get(
            "http://169.254.169.254/metadata/identity/oauth2/token",
            headers={"Metadata": "true"},
            params={"api-version": "2018-02-01", "resource": resource},
            timeout=1.0,
        )
        if r.ok:
            return r.json().get("access_token")
    except Exception:
        pass
    return None

def _auth_headers():
    tok = _get_msi_token()
    if not tok:
        raise RuntimeError("Managed Identity token unavailable. Enable system-assigned identity and grant role on the Foundry project.")
    return {"Authorization": f"Bearer {tok}", "Content-Type": "application/json", "Accept": "application/json"}

# --- helpers ---
def _extract_text(fragment):
    """
    Handles v1 content shapes:
      {"type":"output_text","text":"..."}
      {"type":"output_text","text":{"value":"...","annotations":[...]}}
      {"type":"text","text":"..."}  or same dict shape
    """
    t = fragment.get("text")
    if isinstance(t, str):
        return t
    if isinstance(t, dict):
        if isinstance(t.get("value"), str):
            return t["value"]
        if isinstance(t.get("text"), str):
            return t["text"]
    return ""

# --- Azure Function entrypoint ---
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if not PROJECT_ENDPOINT:
            return _http(500, {"error": "missing_env", "which": "PROJECT_ENDPOINT"})
        if not AGENT_ID:
            return _http(500, {"error": "missing_env", "which": "AGENT_ID"})

        try:
            body = req.get_json()
        except Exception:
            return _http(400, {"error": "invalid_json"})

        prompt = (body or {}).get("prompt", "").strip()
        if not prompt:
            return _http(400, {"error": "missing_field", "field": "prompt"})

        h = _auth_headers()

        # 1) Probe: check assistant exists & permissions OK
        pr = requests.get(f"{PROJECT_ENDPOINT}/assistants/{AGENT_ID}?api-version={API_VERSION}", headers=h, timeout=8)
        if not pr.ok:
            return _http(500, {"stage": "probe_assistant", "status": pr.status_code, "body": pr.text[:1200]})

        # 2) Create thread
        tr = requests.post(f"{PROJECT_ENDPOINT}/threads?api-version={API_VERSION}", headers=h, json={}, timeout=10)
        if not tr.ok:
            return _http(500, {"stage": "create_thread", "status": tr.status_code, "body": tr.text[:1200]})
        thread_id = tr.json().get("id")

        # 3) Add user message
        msg = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        mr = requests.post(f"{PROJECT_ENDPOINT}/threads/{thread_id}/messages?api-version={API_VERSION}", headers=h, json=msg, timeout=10)
        if not mr.ok:
            return _http(500, {"stage": "create_message", "status": mr.status_code, "body": mr.text[:1200], "thread_id": thread_id})

        # 4) Run
        rr = requests.post(f"{PROJECT_ENDPOINT}/threads/{thread_id}/runs?api-version={API_VERSION}", headers=h, json={"assistant_id": AGENT_ID}, timeout=10)
        if not rr.ok:
            return _http(500, {"stage": "create_run", "status": rr.status_code, "body": rr.text[:1200], "thread_id": thread_id})
        run_id = rr.json().get("id")

        # 5) Poll run
        start = time.time(); status = None
        while True:
            gr = requests.get(f"{PROJECT_ENDPOINT}/threads/{thread_id}/runs/{run_id}?api-version={API_VERSION}", headers=h, timeout=10)
            if not gr.ok:
                return _http(500, {"stage": "get_run", "status": gr.status_code, "body": gr.text[:1200], "thread_id": thread_id, "run_id": run_id})
            status = gr.json().get("status")
            if status in ("completed", "failed", "cancelled", "expired"):
                break
            if time.time() - start > 55:
                return _http(504, {"error": "run_timeout", "status": status, "thread_id": thread_id, "run_id": run_id})
            time.sleep(1)

        if status != "completed":
            return _http(500, {"error": "run_not_completed", "status": status, "thread_id": thread_id, "run_id": run_id})

        # 6) Read assistant answer (robust)
        lm = requests.get(
            f"{PROJECT_ENDPOINT}/threads/{thread_id}/messages?api-version={API_VERSION}&order=desc&limit=20",
            headers=h, timeout=10
        )
        if not lm.ok:
            return _http(500, {"stage": "list_messages", "status": lm.status_code, "body": lm.text[:1200], "thread_id": thread_id})

        answer_parts = []
        for item in lm.json().get("data", []):
            if item.get("role") == "assistant":
                for c in (item.get("content") or []):
                    if c.get("type") in ("output_text", "text"):
                        frag = _extract_text(c)
                        if frag:
                            answer_parts.append(frag)
                break

        final_text = ("".join(answer_parts)).strip()

        # --- New logic: try parsing JSON answer and flatten useful fields
        resp = {
            "ok": True,
            "thread_id": thread_id,
            "run_id": run_id
        }

        try:
            parsed = json.loads(final_text)
            if isinstance(parsed, dict):
                # flatten known fields
                resp["status"] = parsed.get("status")
                resp["issue"] = parsed.get("issue")
                resp["summary"] = parsed.get("summary")
                resp["description"] = parsed.get("description")
                resp["ticketUrl"] = parsed.get("ticketUrl")
                resp["assignee"] = parsed.get("assignee")
                resp["confidence"] = parsed.get("confidence")

                # --- keep the adaptiveCard and extract missing fields ---
                if "adaptiveCard" in parsed:
                    resp["adaptiveCard"] = parsed["adaptiveCard"]

                    body = parsed["adaptiveCard"].get("body", [])
                    actions = parsed["adaptiveCard"].get("actions", [])

                    for block in body:
                        if isinstance(block, dict):
                            txt = block.get("text", "")
                            if txt.startswith("**Issue ID:**") and not resp.get("issue"):
                                resp["issue"] = txt.replace("**Issue ID:**", "").strip()
                            elif txt.startswith("**Summary:**") and not resp.get("summary"):
                                resp["summary"] = txt.replace("**Summary:**", "").strip()
                            elif txt.startswith("**Description:**") and not resp.get("description"):
                                resp["description"] = txt.replace("**Description:**", "").strip()

                    if not resp.get("ticketUrl") and actions:
                        for action in actions:
                            if action.get("type") == "Action.OpenUrl":
                                resp["ticketUrl"] = action.get("url")
                                break
            else:
                resp["answer"] = final_text
        except Exception:
            # if not JSON → keep as raw answer
            resp["answer"] = final_text

        return _http(200, resp)

    except Exception as e:
        return _http(500, {"error": "unhandled_exception", "detail": repr(e), "traceback": traceback.format_exc()[:3000]})
