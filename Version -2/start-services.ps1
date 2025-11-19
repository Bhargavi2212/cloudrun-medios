# Start Manage Agent Service
Write-Host "Starting Manage Agent on port 8001..." -ForegroundColor Green
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:DATABASE_URL='postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2'; `$env:GEMINI_API_KEY='AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo'; poetry run uvicorn services.manage_agent.main:app --port 8001 --reload"

# Start DOL Service
Write-Host "Starting DOL Service on port 8004..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; `$env:DATABASE_URL='postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2'; `$env:DOL_SHARED_SECRET='super-secret'; `$env:DOL_HOSPITAL_ID='ORCHESTRATOR_CORE'; poetry run uvicorn dol_service.main:app --port 8004 --reload"

Write-Host "Services starting in separate windows..." -ForegroundColor Yellow
Write-Host "Manage Agent: http://localhost:8001" -ForegroundColor Cyan
Write-Host "DOL Service: http://localhost:8004" -ForegroundColor Cyan

