#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"

echo "Launching services..."
python "$ROOT/services/offer_fake_agent/server.py" & 
python "$ROOT/services/cv_fake_agent/server.py" & 
python "$ROOT/services/capacity_fake_agent/server.py" & 
python "$ROOT/services/finance_fake_agent/server.py" & 
python "$ROOT/services/policy_fake_agent/server.py" & 
python "$ROOT/services/calendar_fake_agent/server.py" & 
python "$ROOT/services/kpi_fake_agent/server.py" & 
python "$ROOT/services/bu_fake_agent/server.py" & 
python "$ROOT/services/jira_fake_agent/server.py" &

echo "All services started (background)."
wait
