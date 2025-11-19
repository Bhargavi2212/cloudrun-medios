# System Ready to Start

## Migration Status: COMPLETE

The database migration has been successfully run. The following tables have been created:
- `file_assets` - For storing uploaded document metadata
- `timeline_events` - For storing extracted timeline events

## Database Credentials

Configure your PostgreSQL database and set the connection string in the `DATABASE_URL` environment variable.

Example connection string format: `postgresql+asyncpg://username:password@localhost:5432/database_name`

## Environment Variables

Set these in each terminal before starting services:

```powershell
$env:DATABASE_URL="postgresql+asyncpg://username:password@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="your-gemini-api-key-here"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage"
```

**Note:** Replace `username`, `password`, and `your-gemini-api-key-here` with your actual credentials.

## Starting Services

### Terminal 1 - Manage Agent (Port 8001)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://username:password@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="your-gemini-api-key-here"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage"
uvicorn services.manage_agent.main:app --port 8001 --reload
```

### Terminal 2 - Summarizer Agent (Port 8003)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2"
$env:DATABASE_URL="postgresql+asyncpg://username:password@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="your-gemini-api-key-here"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage"
uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

### Terminal 3 - Frontend (Port 5173)

```powershell
cd "D:\Hackathons\Cloud Run\Version -2\apps\frontend"
npm run dev
```

## Verify Services

Once services are running, verify they're accessible:

- Manage Agent: http://localhost:8001/health
- Summarizer Agent: http://localhost:8003/health
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

