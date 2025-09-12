Write-Output "Launching services..."

Start-Process powershell -ArgumentList "-NoExit", "python -m services.offer_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.cv_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.capacity_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.finance_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.policy_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.calendar_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.jira_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.kpi_fake_agent.server"
Start-Process powershell -ArgumentList "-NoExit", "python -m services.bu_fake_agent.server"

Write-Output "All services launched in separate windows."
