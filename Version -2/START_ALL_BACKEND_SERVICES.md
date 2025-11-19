# Start All Backend Services - Complete Guide

This guide provides commands to start all backend services for the Medi OS application.

## Services Overview

| Service | Port | Description | Required |
|---------|------|-------------|----------|
| **Manage Agent** | 8001 | Patient management, documents, queue | ✅ Essential |
| **Scribe Agent** | 8002 | SOAP note generation, transcript processing | ✅ Essential |
| **Summarizer Agent** | 8003 | AI summary generation | ✅ Essential |
| **DOL Service** | 8004 | Data Orchestration Layer | ⚠️ Optional |
| **Federation Service** | 8010 | Cross-hospital data sharing | ⚠️ Optional |

---

## Quick Start (PowerShell - All Services)

### Option 1: Automated Script (Recommended)

Run the provided script to start all services in separate windows:

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
.\start-all-backend.ps1
```

### Option 2: Manual Commands

Open **5 separate PowerShell terminals** and run one command in each:

#### Terminal 1: Manage Agent (Port 8001)
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage"
poetry run uvicorn services.manage_agent.main:app --port 8001 --reload
```

#### Terminal 2: Scribe Agent (Port 8002)
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:SCRIBE_AGENT_CORS_ORIGINS='["http://localhost:5173","http://127.0.0.1:5173"]'
poetry run uvicorn services.scribe_agent.main:app --port 8002 --reload
```

#### Terminal 3: Summarizer Agent (Port 8003)
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:SUMMARIZER_AGENT_CORS_ORIGINS='["http://localhost:5173","http://127.0.0.1:5173"]'
poetry run uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

#### Terminal 4: DOL Service (Port 8004) - Optional
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:DOL_SHARED_SECRET="super-secret"
$env:DOL_HOSPITAL_ID="ORCHESTRATOR_CORE"
poetry run uvicorn dol_service.main:app --port 8004 --reload
```

#### Terminal 5: Federation Service (Port 8010) - Optional
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:FEDERATION_SHARED_SECRET="federation-secret"
poetry run uvicorn services.federation_agent.main:app --port 8010 --reload
```

---

## Minimal Setup (Essential Services Only)

If you only need the core functionality, start these 3 services:

### Terminal 1: Manage Agent
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.manage_agent.main:app --port 8001 --reload
```

### Terminal 2: Scribe Agent
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.scribe_agent.main:app --port 8002 --reload
```

### Terminal 3: Summarizer Agent
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

---

## Verification

After starting services, verify they're running:

### Check Ports
```powershell
netstat -ano | findstr ":8001 :8002 :8003 :8004 :8010"
```

### Test Endpoints
```powershell
# Manage Agent
curl http://localhost:8001/health

# Scribe Agent
curl http://localhost:8002/health

# Summarizer Agent
curl http://localhost:8003/summarizer/test

# DOL Service
curl http://localhost:8004/health

# Federation Service
curl http://localhost:8010/health
```

---

## Environment Variables Reference

### Common Variables (All Services)
- `DATABASE_URL`: PostgreSQL connection string
  - Format: `postgresql+asyncpg://user:password@host:port/database`
  - Default: `postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2`

### Service-Specific Variables

#### Manage Agent
- `MANAGE_AGENT_STORAGE_ROOT`: Storage directory for uploaded files (default: `./storage`)

#### Scribe Agent
- `SCRIBE_AGENT_CORS_ORIGINS`: CORS allowed origins (JSON array)
  - Example: `'["http://localhost:5173","http://127.0.0.1:5173"]'`

#### Summarizer Agent
- `SUMMARIZER_AGENT_CORS_ORIGINS`: CORS allowed origins (JSON array)
- `GEMINI_API_KEY`: Google Gemini API key (required for AI features)

#### DOL Service
- `DOL_SHARED_SECRET`: Shared secret for inter-service communication
- `DOL_HOSPITAL_ID`: Hospital identifier

#### Federation Service
- `FEDERATION_SHARED_SECRET`: Secret for federation communication

---

## Troubleshooting

### Port Already in Use
If you get "port already in use" error:

```powershell
# Find process using the port
netstat -ano | findstr ":8001"

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Service Not Starting
1. Check if PostgreSQL is running:
   ```powershell
   # Check if PostgreSQL is running
   Get-Service -Name postgresql*
   ```

2. Verify database connection:
   ```powershell
   # Test connection
   psql -U postgres -d medi_os_v2 -h localhost
   ```

3. Check Poetry environment:
   ```powershell
   poetry env info
   poetry install
   ```

### Services Not Reloading
If code changes aren't being picked up:
1. Stop the service (Ctrl+C)
2. Clear Python cache:
   ```powershell
   Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force
   Get-ChildItem -Path . -Include *.pyc -Recurse -File | Remove-Item -Force
   ```
3. Restart the service

---

## Frontend Setup

After starting backend services, start the frontend:

```powershell
cd "D:\Hackathons\Cloud Run\Version -2\apps\frontend"
npm run dev
```

Frontend will be available at: `http://localhost:5173`

---

## Service URLs

Once all services are running:

| Service | URL | Health Check |
|---------|-----|--------------|
| Manage Agent | http://localhost:8001 | http://localhost:8001/health |
| Scribe Agent | http://localhost:8002 | http://localhost:8002/health |
| Summarizer Agent | http://localhost:8003 | http://localhost:8003/summarizer/test |
| DOL Service | http://localhost:8004 | http://localhost:8004/health |
| Federation Service | http://localhost:8010 | http://localhost:8010/health |
| Frontend | http://localhost:5173 | http://localhost:5173 |

---

## Quick Reference Card

```powershell
# Essential Services (3 terminals)
# Terminal 1
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.manage_agent.main:app --port 8001 --reload

# Terminal 2
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.scribe_agent.main:app --port 8002 --reload

# Terminal 3
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
poetry run uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

---

## Notes

- All services use `--reload` flag for automatic code reloading during development
- Services are independent and can be started in any order
- Database must be running before starting services
- Gemini API key is required for AI features (summarizer, scribe)
- CORS origins are only needed if accessing from frontend

