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
$env:DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2"
$env:GEMINI_API_KEY="your-gemini-api-key-here"
$env:MANAGE_AGENT_STORAGE_ROOT="./storage"
$env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage"
```

**Linux/Mac:**
```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/medi_os_v2"
export GEMINI_API_KEY="your-gemini-api-key-here"
export MANAGE_AGENT_STORAGE_ROOT="./storage"
export SUMMARIZER_AGENT_STORAGE_ROOT="./storage"
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
uvicorn services.manage_agent.main:app --port 8001 --reload
```

**Terminal 2 - Summarizer Agent:**
```bash
cd "Version -2"
uvicorn services.summarizer_agent.main:app --port 8003 --reload
```

Verify services are running:
- http://localhost:8001/health
- http://localhost:8003/health

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

**Manage Agent (port 8001):**
- POST /manage/documents/upload - Upload document
- GET /manage/documents - List all documents
- GET /manage/documents/{file_id} - Get document details
- GET /manage/documents/pending-review - List pending documents
- POST /manage/documents/{file_id}/confirm - Confirm document
- POST /manage/documents/{file_id}/reject - Reject document

**Summarizer Agent (port 8003):**
- POST /summarizer/documents/{file_id}/process - Process document

## Troubleshooting

**If services fail to start:**
- Check DATABASE_URL is set correctly
- Verify database is running and accessible
- Check port 8001 and 8003 are not in use

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
curl http://localhost:8001/health
curl http://localhost:8003/health
```

