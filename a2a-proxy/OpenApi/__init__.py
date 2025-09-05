import json
import azure.functions as func

OPENAPI = {
    "openapi": "3.0.3",
    "info": {
        "title": "Azure Agent Proxy",
        "version": "1.0.0"
    },
    "paths": {
        "/ask": {
            "post": {
                "summary": "Send a prompt to the Azure Agent and get its answer",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["prompt"],
                                "properties": {
                                    "prompt": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "ok": {"type": "boolean"},
                                        "thread_id": {"type": ["string", "null"]},
                                        "run_id": {"type": ["string", "null"]},
                                        "status": {"type": ["string", "null"]},
                                        "issue": {"type": ["string", "null"]},
                                        "summary": {"type": ["string", "null"]},
                                        "description": {"type": ["string", "null"]},
                                        "ticketUrl": {"type": ["string", "null"]},
                                        "assignee": {"type": ["string", "null"]},
                                        "confidence": {"type": ["number", "null"]},
                                        "answer": {"type": ["string", "null"]},
                                        "adaptiveCard": {
                                            "type": "object",
                                            "x-nullable": True,
                                            "properties": {
                                                "type": {"type": "string"},
                                                "version": {"type": "string"},
                                                "body": {
                                                    "type": "array",
                                                    "items": {"type": "object"}
                                                },
                                                "actions": {
                                                    "type": "array",
                                                    "items": {"type": "object"}
                                                }
                                            },
                                            "description": "AdaptiveCard JSON if present"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "servers": [{"url": "/"}]
}


def main(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(OPENAPI, indent=2),
        status_code=200,
        mimetype="application/json"
    )
