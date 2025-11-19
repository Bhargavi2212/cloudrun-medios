#!/bin/bash
# Pre-push validation script
# Runs all linting and tests before pushing to Git

set -e  # Exit on any error

echo "=========================================="
echo "Medi OS Pre-Push Validation"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED=0

# Function to run command and track failures
run_check() {
    local name=$1
    shift
    echo -e "${YELLOW}[RUN] $name${NC}"
    if "$@"; then
        echo -e "${GREEN}[PASS] $name${NC}\n"
    else
        echo -e "${RED}[FAIL] $name${NC}\n"
        FAILED=$((FAILED + 1))
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from Version -2 directory${NC}"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed. Install with: pip install poetry${NC}"
    exit 1
fi

# Check if Node/npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

echo "Checking dependencies..."
echo ""

# Install/update dependencies if needed
if [ ! -d ".venv" ] || [ ! -d "apps/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    poetry install
    cd apps/frontend && npm install && cd ../..
    echo ""
fi

echo "=========================================="
echo "Backend Checks"
echo "=========================================="
echo ""

# Backend linting
run_check "Ruff linting" poetry run ruff check .
run_check "Black format check" poetry run black --check .
run_check "MyPy type checking" poetry run mypy . || true  # MyPy might have some issues, but we'll still check

echo "=========================================="
echo "Backend Tests"
echo "=========================================="
echo ""

# Check if DATABASE_URL is set for tests
if [ -z "$DATABASE_URL" ] && [ -z "$TEST_DATABASE_URL" ]; then
    echo -e "${YELLOW}[WARN] DATABASE_URL not set. Some tests may be skipped.${NC}"
    echo -e "${YELLOW}   Set DATABASE_URL for full test coverage.${NC}\n"
fi

run_check "Pytest" poetry run pytest -v --tb=short || true  # Continue even if some tests fail

echo ""
echo "=========================================="
echo "Frontend Checks"
echo "=========================================="
echo ""

cd apps/frontend

# Frontend linting
run_check "ESLint" npm run lint -- --max-warnings=0

# Frontend type checking
run_check "TypeScript compilation" npx tsc --noEmit

echo ""
echo "=========================================="
echo "Frontend Tests"
echo "=========================================="
echo ""

run_check "Vitest" npm run test -- --run

cd ../..

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}[PASS] All checks passed! Ready to push.${NC}"
    exit 0
else
    echo -e "${RED}[FAIL] $FAILED check(s) failed. Please fix issues before pushing.${NC}"
    exit 1
fi

