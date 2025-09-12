# orchestrator/agent.py
from google.adk.agents import Agent
from agents.nvidia_orchestrator_tool import nvidia_orchestrate

demo_director = Agent(
    name="demo_director",
    model="gemini-1.5-flash-8b",  # tiny planner call, not the heavy lifting
    instruction=(
        "Appelle UNE SEULE FOIS la fonction nvidia_orchestrate(input) avec "
        "exactement le texte utilisateur, puis ARRÊTE. "
        "N’écris AUCUN texte libre, ne génère PAS de code, ne fais PAS d’autres appels."
    ),
    tools=[nvidia_orchestrate],
)

root_agent = demo_director
