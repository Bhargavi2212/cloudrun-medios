# Start Essential Backend Services Only
# This script starts only the 3 essential services (Manage, Scribe, Summarizer)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Essential Backend Services" -ForegroundColor Cyan
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

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Essential Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service URLs:" -ForegroundColor Yellow
Write-Host "  Manage Agent:      http://localhost:8001" -ForegroundColor White
Write-Host "  Scribe Agent:      http://localhost:8002" -ForegroundColor White
Write-Host "  Summarizer Agent:  http://localhost:8003" -ForegroundColor White
Write-Host ""
Write-Host "Each service is running in a separate PowerShell window." -ForegroundColor Cyan
Write-Host "Close the windows or press Ctrl+C to stop services." -ForegroundColor Yellow
Write-Host ""

