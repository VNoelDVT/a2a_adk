# orchestrator/agent.py

from google.adk.agents import Agent
from agents.intake_normalizer_agent import detect_and_normalize
from agents.offer_vectorizer_agent import make_offer_vector
from agents.candidate_matcher_agent import match_candidates
from agents.capacity_checker_agent import check_capacity_for_list
from agents.finance_policy_agent import enrich_finance_policy
from agents.explainability_agent import explain_shortlist
from agents.geo_filter_agent import filter_by_geo
from agents.bu_health_agent import bu_overview
from orchestrator.ui_contracts import shortlist_view, bu_health_view

demo_director = Agent(
    name="demo_director",
    model="gemini-2.0-flash-001",
    instruction=(
        "Tu es l’orchestrateur. Tu enchaînes des APPELS D’OUTILS pour réaliser la tâche.\n\n"
        "RÈGLE FINALE: après avoir appelé exactement une fois `shortlist_view(...)` "
        "ou `bu_health_view(...)`, tu DOIS faire un dernier tour en TEXTE LIBRE, "
        "sans aucun autre appel d’outil, et IMPRIMER EXCLUSIVEMENT le champ `text` "
        "renvoyé par cet outil, tel quel, puis t’arrêter.\n\n"

        "### Cas OFFRE -> CANDIDATS\n"
        "1) detect_and_normalize(input)\n"
        "2) make_offer_vector(offer)\n"
        "3) match_candidates(offer_vector)\n"
        "4) check_capacity_for_list(candidates, offer.start_by)\n"
        "5) filter_by_geo({\"shortlist\": shortlist, \"location\": offer.location, \"max_km\": 400})\n"
        "6) enrich_finance_policy(offer, shortlist)\n"
        "7) (optionnel) explain_shortlist(offer, shortlist)\n"
        "8) shortlist_view(offer, shortlist)  # puis dernier tour EN TEXTE avec le champ `text` de la sortie\n\n"

        "### Cas CONSULTANT -> MISSIONS\n"
        "1) detect_and_normalize(input)\n"
        "2) match_candidates(vector, target='offers')\n"
        "3) check_capacity_for_list(offers, consultant.availability.from)\n"
        "4) filter_by_geo({\"shortlist\": offers, \"location\": consultant.location, \"max_km\": 400})\n"
        "5) enrich_finance_policy({\"budget_tjm\": 9999}, offers)\n"
        "6) shortlist_view({\"role\": \"MATCHES\", \"client_id\": \"-\"}, offers)  # puis dernier tour EN TEXTE avec `text`\n\n"

        "### Cas BU\n"
        "1) detect_and_normalize(input)\n"
        "2) bu_overview(bu_name)\n"
        "3) bu_health_view(report)  # puis dernier tour EN TEXTE avec `text`\n"
    ),
    tools=[
        detect_and_normalize,
        make_offer_vector,
        match_candidates,
        check_capacity_for_list,
        filter_by_geo,
        enrich_finance_policy,
        explain_shortlist,
        shortlist_view,
        bu_overview,
        bu_health_view,
    ],
)

root_agent = demo_director
