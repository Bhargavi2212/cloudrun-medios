# PowerShell script to start all services with correct credentials

Write-Host "Setting up environment..." -ForegroundColor Green

# Database credentials - IMPORTANT: Set these as environment variables, not hardcoded!
# NEVER commit actual passwords or API keys to git!
if (-not $env:DATABASE_URL) {
    $dbPassword = Read-Host -Prompt "Enter PostgreSQL password" -AsSecureString
    $dbPasswordPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($dbPassword))
    $env:DATABASE_URL = "postgresql+asyncpg://postgres:$dbPasswordPlain@localhost:5432/medi_os_v2_b"
}

if (-not $env:GEMINI_API_KEY) {
    Write-Warning "GEMINI_API_KEY not set. Document processing will be disabled."
    $env:GEMINI_API_KEY = ""
}
$env:MANAGE_AGENT_STORAGE_ROOT = "./storage/hospital_b"
$env:SUMMARIZER_AGENT_STORAGE_ROOT = "./storage/hospital_b"

Write-Host "Environment variables set:" -ForegroundColor Yellow
Write-Host "  DATABASE_URL: Set" -ForegroundColor Green
Write-Host "  GEMINI_API_KEY: Set" -ForegroundColor Green
Write-Host "  Storage roots: Set" -ForegroundColor Green
Write-Host ""

Write-Host "To start services, run in separate terminals:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Terminal 1 - Manage Agent:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2"' -ForegroundColor Gray
Write-Host '  $env:DATABASE_URL="postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/medi_os_v2_b"' -ForegroundColor Gray
Write-Host '  $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"' -ForegroundColor Gray
Write-Host '  uvicorn services.manage_agent.main:app --port 9001 --reload' -ForegroundColor Yellow
Write-Host ""
Write-Host "Terminal 2 - Summarizer Agent:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2"' -ForegroundColor Gray
Write-Host '  $env:DATABASE_URL="postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/medi_os_v2_b"' -ForegroundColor Gray
Write-Host '  $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"' -ForegroundColor Gray
Write-Host '  uvicorn services.summarizer_agent.main:app --port 9003 --reload' -ForegroundColor Yellow
Write-Host ""
Write-Host "Terminal 3 - Frontend:" -ForegroundColor White
Write-Host '  cd "D:\Hackathons\Cloud Run\Version -2\apps\frontend"' -ForegroundColor Gray
Write-Host '  npm run dev' -ForegroundColor Yellow
Write-Host ""
Write-Host "Then navigate to: http://localhost:5173/documents" -ForegroundColor Cyan

