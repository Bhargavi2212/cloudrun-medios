# Document System Testing Guide

## Prerequisites

1. Database migration must be run first:
   ```bash
   cd "Version -2"
   alembic upgrade head
   ```

2. Install Python dependencies:
   ```bash
   pip install -r services/manage_agent/requirements.txt
   pip install -r services/summarizer_agent/requirements.txt
   ```

3. Set environment variables:
   - `DATABASE_URL` - PostgreSQL connection string (hospital B)
   - `GEMINI_API_KEY` - Google Gemini API key (required for document processing)
   - `MANAGE_AGENT_STORAGE_ROOT` - Storage directory (default: ./storage/hospital_b)
   - `SUMMARIZER_AGENT_STORAGE_ROOT` - Must match manage-agent storage root

   Quick setup (PowerShell):
   ```powershell
   $env:DATABASE_URL="postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/medi_os_v2_b"
   $env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
   $env:MANAGE_AGENT_STORAGE_ROOT="./storage/hospital_b"
   $env:SUMMARIZER_AGENT_STORAGE_ROOT="./storage/hospital_b"
   ```

   See SETUP_ENV.md for detailed instructions.

## Testing Steps

### 1. Verify Imports
```bash
cd "Version -2"
python test_document_system.py
```

Expected output: All [OK] messages

### 2. Start Backend Services

Terminal 1 - Manage Agent:
```bash
cd "Version -2"
uvicorn services.manage_agent.main:app --port 9001 --reload
```

Terminal 2 - Summarizer Agent:
```bash
cd "Version -2"
uvicorn services.summarizer_agent.main:app --port 9003 --reload
```

### 3. Test API Endpoints

#### Upload Document
```bash
curl -X POST "http://localhost:9001/manage/documents/upload" \
  -F "file=@test.pdf" \
  -F "patient_id=<patient-uuid>" \
  -F "upload_method=file_picker"
```

#### List Documents
```bash
curl "http://localhost:9001/manage/documents"
```

#### List Pending Documents
```bash
curl "http://localhost:9001/manage/documents/pending-review"
```

#### Process Document
```bash
curl -X POST "http://localhost:9003/summarizer/documents/<file-id>/process"
```

#### Confirm Document
```bash
curl -X POST "http://localhost:9001/manage/documents/<file-id>/confirm" \
  -F "notes=Approved by nurse"
```

#### Reject Document
```bash
curl -X POST "http://localhost:9001/manage/documents/<file-id>/reject" \
  -F "reason=Poor quality scan"
```

### 4. Start Frontend

```bash
cd "Version -2/apps/frontend"
npm install
npm run dev
```

Navigate to: http://localhost:5173/documents

### 5. Test Frontend Flow

1. Go to Documents page
2. Click "Upload Documents" tab
3. Upload a PDF or image file
4. File should appear in "Review Documents" tab after processing
5. Click "View Details" to see extracted data
6. Click "Approve" or "Reject" to complete review

## Expected Behavior

1. Upload creates FileAsset record with status "uploaded"
2. Processing extracts text, runs 4-step Gemini pipeline
3. Creates TimelineEvent if confidence >= 75%
4. Documents with confidence < 75% require manual review
5. Nurse can confirm/reject documents in review dashboard

## Troubleshooting

- If Gemini API errors: Check GEMINI_API_KEY is set correctly
- If storage errors: Check storage_root directory exists and is writable
- If database errors: Verify migration ran successfully
- If import errors: Check Python path includes Version -2 directory

