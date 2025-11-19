# Quick Start - Backend Services

## 🚀 Fastest Way to Start

### Option 1: Automated Script (Recommended)
```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
.\start-essential-services.ps1    # Starts 3 essential services
# OR
.\start-all-backend.ps1            # Starts all 5 services
```

### Option 2: Copy-Paste Commands

**Essential Services (3 terminals):**

```powershell
# Terminal 1 - Manage Agent
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
poetry run uvicorn services.manage_agent.main:app --port 8001 --reload
```

```powershell
# Terminal 2 - Scribe Agent
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:SCRIBE_AGENT_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
poetry run uvicorn services.scribe_agent.main:app --port 8002 --reload
```

```powershell
# Terminal 3 - Summarizer Agent
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:SUMMARIZER_AGENT_CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
poetry run uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

## ✅ Verify Services Are Running

```powershell
# Check ports
netstat -ano | findstr ":8001 :8002 :8003"

# Test endpoints
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/summarizer/test
```

## 📋 Service Ports

- **8001** - Manage Agent (Patient management, documents)
- **8002** - Scribe Agent (SOAP notes, transcripts)
- **8003** - Summarizer Agent (AI summaries)
- **8004** - DOL Service (Optional)
- **8010** - Federation Service (Optional)

## 🛑 Stop Services

Press `Ctrl+C` in each terminal window, or close the windows.

## 📚 Full Documentation

See `START_ALL_BACKEND_SERVICES.md` for complete details.

