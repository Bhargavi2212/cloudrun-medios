# Start All Backend Services
# This script starts all backend services in separate PowerShell windows

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting All Backend Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Common environment variables
$dbUrl = "postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$geminiKey = "AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"

# Terminal 1: Manage Agent (Port 8001)
Write-Host "Starting Manage Agent on port 8001..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:MANAGE_AGENT_STORAGE_ROOT='./storage'
`$env:MANAGE_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Manage Agent starting on port 8001...' -ForegroundColor Cyan
poetry run uvicorn services.manage_agent.main:app --port 8001 --reload
"@

Start-Sleep -Seconds 2

# Terminal 2: Scribe Agent (Port 8002)
Write-Host "Starting Scribe Agent on port 8002..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:SCRIBE_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Scribe Agent starting on port 8002...' -ForegroundColor Cyan
poetry run uvicorn services.scribe_agent.main:app --port 8002 --reload
"@

Start-Sleep -Seconds 2

# Terminal 3: Summarizer Agent (Port 8003)
Write-Host "Starting Summarizer Agent on port 8003..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:GEMINI_API_KEY='$geminiKey'
`$env:SUMMARIZER_AGENT_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
Write-Host 'Summarizer Agent starting on port 8003...' -ForegroundColor Cyan
poetry run uvicorn services.summarizer_agent.main:app --port 8003 --reload
"@

Start-Sleep -Seconds 2

# Terminal 4: DOL Service (Port 8004) - Optional
Write-Host "Starting DOL Service on port 8004 (Optional)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:DOL_SHARED_SECRET='super-secret'
`$env:DOL_HOSPITAL_ID='ORCHESTRATOR_CORE'
Write-Host 'DOL Service starting on port 8004...' -ForegroundColor Cyan
poetry run uvicorn dol_service.main:app --port 8004 --reload
"@

Start-Sleep -Seconds 2

# Terminal 5: Federation Service (Port 8010) - Optional
Write-Host "Starting Federation Service on port 8010 (Optional)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
cd '$scriptPath'
`$env:DATABASE_URL='$dbUrl'
`$env:FEDERATION_SHARED_SECRET='federation-secret'
Write-Host 'Federation Service starting on port 8010...' -ForegroundColor Cyan
poetry run uvicorn services.federation_agent.main:app --port 8010 --reload
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Manage Agent:      http://localhost:8001" -ForegroundColor White
Write-Host "  Scribe Agent:      http://localhost:8002" -ForegroundColor White
Write-Host "  Summarizer Agent:  http://localhost:8003" -ForegroundColor White
Write-Host "  DOL Service:       http://localhost:8004" -ForegroundColor Gray
Write-Host "  Federation Service: http://localhost:8010" -ForegroundColor Gray
Write-Host ""
Write-Host "Each service is running in a separate PowerShell window." -ForegroundColor Cyan
Write-Host "Close the windows or press Ctrl+C to stop services." -ForegroundColor Yellow
Write-Host ""

