# PowerShell script to start all services with correct credentials

Write-Host "Setting up environment..." -ForegroundColor Green

# Database credentials
$env:DATABASE_URL = "postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_a"
$env:GEMINI_API_KEY = "AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT = "./storage/hospital_a"
$env:SUMMARIZER_AGENT_STORAGE_ROOT = "./storage/hospital_a"

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  DATABASE_URL: Set" -ForegroundColor Green
Write-Host "  GEMINI_API_KEY: Set" -ForegroundColor Green
Write-Host "  Storage roots: Set" -ForegroundColor Green
Write-Host ""

Write-Host "To start services, run in separate terminals:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1 - Manage Agent:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2"' -ForegroundColor Gray
Write-Host '  $env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_a"' -ForegroundColor Gray
Write-Host '  $env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"' -ForegroundColor Gray
Write-Host '  uvicorn services.manage_agent.main:app --port 8001 --reload' -ForegroundColor Yellow
Write-Host ""
Write-Host "Terminal 2 - Summarizer Agent:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2"' -ForegroundColor Gray
Write-Host '  $env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_a"' -ForegroundColor Gray
Write-Host '  $env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"' -ForegroundColor Gray
Write-Host '  uvicorn services.summarizer_agent.main:app --port 8003 --reload' -ForegroundColor Yellow
Write-Host ""
Write-Host "Terminal 3 - Frontend:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2\apps\frontend"' -ForegroundColor Gray
Write-Host '  npm run dev' -ForegroundColor Yellow
Write-Host ""
Write-Host "Then navigate to: http://localhost:5173/documents" -ForegroundColor Cyan

