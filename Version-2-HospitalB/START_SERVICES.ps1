# PowerShell script to start all services for document system

Write-Host "Starting Document System Services..." -ForegroundColor Green

# Set environment variables
$env:GEMINI_API_KEY = "AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT = "./storage/hospital_b"
$env:SUMMARIZER_AGENT_STORAGE_ROOT = "./storage/hospital_b"

Write-Host "Environment variables set" -ForegroundColor Yellow
Write-Host "GEMINI_API_KEY: Set" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: Set DATABASE_URL before starting services:" -ForegroundColor Red
Write-Host '  $env:DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_b"' -ForegroundColor Yellow
Write-Host ""
Write-Host "To start services, run in separate terminals:" -ForegroundColor Cyan
Write-Host "  Terminal 1: uvicorn services.manage_agent.main:app --port 9001 --reload" -ForegroundColor White
Write-Host "  Terminal 2: uvicorn services.summarizer_agent.main:app --port 9003 --reload" -ForegroundColor White
Write-Host "  Terminal 3: cd apps/frontend && npm run dev" -ForegroundColor White
Write-Host ""

