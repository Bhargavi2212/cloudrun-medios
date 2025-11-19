# Start Hospital B (County Hospital) Backend Services
# This script starts Hospital B services on ports 8011, 8012, 8013

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Hospital B (County Hospital) Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Common environment variables
# IMPORTANT: Hospital B uses a SEPARATE database
$dbUrl = "postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_hospital_b"
$geminiKey = "AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"

# Terminal 1: Hospital B - Manage Agent (Port 8011)
Write-Host "Starting Hospital B Manage Agent on port 8011..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:MANAGE_AGENT_STORAGE_ROOT='./storage'
`$env:MANAGE_AGENT_HOSPITAL_ID='hospital-b'
`$env:MANAGE_AGENT_HOSPITAL_NAME='County Hospital'
`$env:ORCHESTRATOR_BASE_URL='http://localhost:8004'
`$env:ORCHESTRATOR_SHARED_SECRET='super-secret'
`$env:MANAGE_AGENT_DOL_BASE_URL='http://localhost:8004'
`$env:MANAGE_AGENT_DOL_SECRET='super-secret'
`$env:MANAGE_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Hospital B - Manage Agent starting on port 8011...' -ForegroundColor Cyan
Write-Host 'Hospital ID: hospital-b' -ForegroundColor Yellow
poetry run uvicorn services.manage_agent.main:app --port 8011 --reload
"@

Start-Sleep -Seconds 2

# Terminal 2: Hospital B - Scribe Agent (Port 8012)
Write-Host "Starting Hospital B Scribe Agent on port 8012..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:SCRIBE_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Hospital B - Scribe Agent starting on port 8012...' -ForegroundColor Cyan
poetry run uvicorn services.scribe_agent.main:app --port 8012 --reload
"@

Start-Sleep -Seconds 2

# Terminal 3: Hospital B - Summarizer Agent (Port 8013)
Write-Host "Starting Hospital B Summarizer Agent on port 8013..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:SUMMARIZER_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Hospital B - Summarizer Agent starting on port 8013...' -ForegroundColor Cyan
poetry run uvicorn services.summarizer_agent.main:app --port 8013 --reload
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Hospital B Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  - Manage Agent:  http://localhost:8011" -ForegroundColor White
Write-Host "  - Scribe Agent:  http://localhost:8012" -ForegroundColor White
Write-Host "  - Summarizer Agent: http://localhost:8013" -ForegroundColor White
Write-Host ""
Write-Host "Verify services are running:" -ForegroundColor Yellow
Write-Host "  netstat -ano | findstr ':8011 :8012 :8013'" -ForegroundColor Gray
Write-Host ""

