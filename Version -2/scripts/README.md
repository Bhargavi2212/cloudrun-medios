# Pre-Push Validation Scripts

These scripts run all linting and tests before pushing to Git to ensure code quality.

## Usage

### Windows (PowerShell)
```powershell
cd "Version -2"
.\scripts\pre-push-checks.ps1
```

### Linux/Mac (Bash)
```bash
cd "Version -2"
chmod +x scripts/pre-push-checks.sh
./scripts/pre-push-checks.sh
```

### Cross-Platform (Python)
```bash
cd "Version -2"
python scripts/pre-push-checks.py
```

## What It Checks

### Backend
- ✅ **Ruff linting** - Python code quality
- ✅ **Black format check** - Code formatting
- ✅ **MyPy type checking** - Type safety (non-blocking)
- ✅ **Pytest** - Unit and integration tests

### Frontend
- ✅ **ESLint** - TypeScript/React code quality
- ✅ **TypeScript compilation** - Type checking
- ✅ **Vitest** - Unit tests

## Prerequisites

1. **Poetry** installed: `pip install poetry`
2. **Node.js 18+** and npm installed
3. **Dependencies installed**: 
   - Backend: `poetry install`
   - Frontend: `cd apps/frontend && npm install`

## Environment Variables

For full test coverage, set:
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2"
export TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_test"
```

## Exit Codes

- `0` - All checks passed ✅
- `1` - One or more checks failed ❌

## Integration with Git

You can integrate this as a Git pre-push hook:

```bash
# Create pre-push hook
cp scripts/pre-push-checks.sh .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

Or run manually before pushing:
```bash
./scripts/pre-push-checks.sh && git push
```

