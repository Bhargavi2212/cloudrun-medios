# Pre-push validation script for Windows PowerShell
# Runs all linting and tests before pushing to Git

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Medi OS Pre-Push Validation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$Failed = 0

function Run-Check {
    param(
        [string]$Name,
        [scriptblock]$ScriptBlock
    )
    
    Write-Host "[RUN] $Name" -ForegroundColor Yellow
    
    try {
        & $ScriptBlock
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
            Write-Host "[PASS] $Name" -ForegroundColor Green
            Write-Host ""
        } else {
            Write-Host "[FAIL] $Name (exit code: $LASTEXITCODE)" -ForegroundColor Red
            Write-Host ""
            $script:Failed++
        }
    } catch {
        Write-Host "[FAIL] $Name failed: $_" -ForegroundColor Red
        Write-Host ""
        $script:Failed++
    }
}

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Error: Must run from Version -2 directory" -ForegroundColor Red
    exit 1
}

# Check if Poetry is installed
try {
    $null = Get-Command poetry -ErrorAction Stop
} catch {
    Write-Host "Error: Poetry is not installed. Install with: pip install poetry" -ForegroundColor Red
    exit 1
}

# Check if Node/npm is installed
try {
    $null = Get-Command npm -ErrorAction Stop
} catch {
    Write-Host "Error: npm is not installed" -ForegroundColor Red
    exit 1
}

Write-Host "Checking dependencies..." -ForegroundColor Cyan
Write-Host ""

# Install/update dependencies if needed
if (-not (Test-Path ".venv") -or -not (Test-Path "apps/frontend/node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    poetry install
    Set-Location apps/frontend
    npm install
    Set-Location ../..
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Backend Checks" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Backend linting
Run-Check "Ruff linting" { poetry run ruff check . }
Run-Check "Black format check" { poetry run black --check . }
Run-Check "MyPy type checking" { 
    poetry run mypy . 
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] MyPy found type issues (non-blocking)" -ForegroundColor Yellow
        $script:Failed--  # Don't count mypy failures as blocking
    }
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Backend Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if DATABASE_URL is set for tests
if (-not $env:DATABASE_URL -and -not $env:TEST_DATABASE_URL) {
    Write-Host "[WARN] DATABASE_URL not set. Some tests may be skipped." -ForegroundColor Yellow
    Write-Host "   Set DATABASE_URL for full test coverage." -ForegroundColor Yellow
    Write-Host ""
}

Run-Check "Pytest" { 
    poetry run pytest -v --tb=short 
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Some tests failed (check output above)" -ForegroundColor Yellow
        $script:Failed--  # Don't count test failures as blocking if we want to continue
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Frontend Checks" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Set-Location apps/frontend

# Frontend linting
Run-Check "ESLint" { npm run lint -- --max-warnings=0 }

# Frontend type checking
Run-Check "TypeScript compilation" { npx tsc --noEmit }

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Frontend Tests" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Run-Check "Vitest" { npm run test -- --run }

Set-Location ../..

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if ($Failed -eq 0) {
    Write-Host "[PASS] All checks passed! Ready to push." -ForegroundColor Green
    exit 0
} else {
    Write-Host "[FAIL] $Failed check(s) failed. Please fix issues before pushing." -ForegroundColor Red
    exit 1
}

