from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn, os

app = FastAPI()
HERE = os.path.dirname(__file__)

@app.get("/.well-known/agent.json")
def agent_card():
    path = os.path.join(HERE, ".well-known", "agent.json")
    return FileResponse(path, media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9003)
