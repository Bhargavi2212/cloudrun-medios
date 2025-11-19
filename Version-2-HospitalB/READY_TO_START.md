# System Ready to Start

## Migration Status: COMPLETE

The database migration has been successfully run. The following tables have been created:
- `file_assets` - For storing uploaded document metadata
- `timeline_events` - For storing extracted timeline events

## Database Credentials

- Username: `postgres`
- Password: `Anuradha`
- Database: `medi_os_v2_b`
- Connection String: `postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b`

## Environment Variables

Set these in each terminal before starting services:

```powershell
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage/hospital_b"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage/hospital_b"
```

## Starting Services

### Terminal 1 - Manage Agent (Port 9001)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage/hospital_b"
uvicorn services.manage_agent.main:app --port 9001 --reload
```

### Terminal 2 - Summarizer Agent (Port 9003)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://postgres:Anuradha@localhost:5432/medi_os_v2_b"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage/hospital_b"
uvicorn services.summarizer_agent.main:app --port 9003 --reload
```

### Terminal 3 - Frontend (Port 5173)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2\apps\frontend"
npm run dev
```

## Verify Services

Once services are running, verify they're accessible:

- Manage Agent: http://localhost:9001/health
- Summarizer Agent: http://localhost:9003/health
- Frontend: http://localhost:5173

## Test the System

1. Navigate to http://localhost:5173/documents
2. Click "Upload Documents" tab
3. Upload a PDF or image file
4. Switch to "Review Documents" tab
5. View extracted data and approve/reject

## API Endpoints

**Manage Agent:**
- POST /manage/documents/upload
- GET /manage/documents
- GET /manage/documents/{file_id}
- GET /manage/documents/pending-review
- POST /manage/documents/{file_id}/confirm
- POST /manage/documents/{file_id}/reject

**Summarizer Agent:**
- POST /summarizer/documents/{file_id}/process

## Status

- [OK] Database migration completed
- [OK] Tables created (file_assets, timeline_events)
- [OK] Services can start successfully
- [OK] Gemini API configured
- [OK] Frontend compiled

**READY TO START TESTING**

