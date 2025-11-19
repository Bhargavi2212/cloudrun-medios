# Quick Start Guide - Document System

## Prerequisites Check

Run this to verify everything is ready:
```bash
cd "Version -2"
python test_full_system.py
```

## Step 1: Set Environment Variables

**PowerShell:**
```powershell
$env:DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_b"
$env:GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage/hospital_b"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage/hospital_b"
```

**Linux/Mac:**
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2_b"
export GEMINI_API_KEY="AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo"
export MANAGE_AGENT_STORAGE_ROOT="./storage/hospital_b"
export SUMMARIZER_AGENT_STORAGE_ROOT="./storage/hospital_b"
```

## Step 2: Run Database Migration

```bash
cd "Version -2"
alembic upgrade head
```

This will create the `file_assets` and `timeline_events` tables.

## Step 3: Start Backend Services

**Terminal 1 - Manage Agent:**
```bash
cd "Version -2"
uvicorn services.manage_agent.main:app --port 9001 --reload
```

**Terminal 2 - Summarizer Agent:**
```bash
cd "Version -2"
uvicorn services.summarizer_agent.main:app --port 9003 --reload
```

Verify services are running:
- http://localhost:9001/health
- http://localhost:9003/health

## Step 4: Start Frontend

**Terminal 3:**
```bash
cd "Version -2/apps/frontend"
npm run dev
```

Frontend will be available at: http://localhost:5173

## Step 5: Test the System

1. Navigate to http://localhost:5173/documents
2. Click "Upload Documents" tab
3. Upload a PDF or image file
4. Switch to "Review Documents" tab
5. View extracted data and approve/reject

## API Endpoints Available

**Manage Agent (port 9001):**
- POST /manage/documents/upload - Upload document
- GET /manage/documents - List all documents
- GET /manage/documents/{file_id} - Get document details
- GET /manage/documents/pending-review - List pending documents
- POST /manage/documents/{file_id}/confirm - Confirm document
- POST /manage/documents/{file_id}/reject - Reject document

**Summarizer Agent (port 9003):**
- POST /summarizer/documents/{file_id}/process - Process document

## Troubleshooting

**If services fail to start:**
- Check DATABASE_URL is set correctly
- Verify database is running and accessible
- Check port 9001 and 9003 are not in use

**If document processing fails:**
- Verify GEMINI_API_KEY is set
- Check google-generativeai package is installed: `pip install google-generativeai==0.3.2`
- Check API key is valid

**If frontend can't connect:**
- Verify backend services are running
- Check CORS settings in service configs
- Verify API URLs in frontend .env file

## Verification Commands

```bash
# Check imports
python test_document_system.py

# Check full system
python test_full_system.py

# Verify migration
alembic current

# Check services
curl http://localhost:9001/health
curl http://localhost:9003/health
```

