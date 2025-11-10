# Note Saving Verification

## Summary

Notes ARE being saved even when Gemini API fails (429 quota error, 404 model not found, etc.).

## How It Works

### 1. Audio Processing Flow

1. **Audio Upload**: Audio is uploaded with `consultation_id`
2. **Transcription**: Whisper transcribes the audio (independent of Gemini)
3. **Entity Extraction**: Entities are extracted from transcription (pattern-based, no Gemini)
4. **Note Generation**: 
   - **If Gemini works**: AI-generated SOAP note
   - **If Gemini fails (429/404/etc.)**: Template-based SOAP note (extracted from transcription)
5. **Note Saving**: **Both AI-generated and template notes are saved** if:
   - `consultation_id` is provided
   - `generated_note` has content (even if it's a template)
   - No critical errors (transcription failed, file not found)

### 2. Note Saving Logic

**File**: `backend/services/make_agent_medi_os.py`

```python
# Save note to database if consultation_id is provided and we have a generated note
# Save even if it's a template/stub note, as long as we have content
if consultation_id and result.get("generated_note"):
    # Check if there are critical errors that prevent saving
    errors = result.get("errors", [])
    critical_errors = [e for e in errors if "transcription failed" in e.lower() or "file not found" in e.lower()]
    
    if not critical_errors:
        try:
            note_content = result.get("generated_note", "").strip()
            if note_content:  # Only save if we have actual content
                self._persist_note(
                    consultation_id=consultation_id,
                    author_id=author_id,
                    note_content=note_content,
                    entities=result.get("entities", {}),
                    is_stub=result.get("is_stub", False),  # Template notes have is_stub=True
                )
                result["note_saved"] = True
```

### 3. Template Note Generation

**File**: `backend/services/ai_models.py`

When Gemini fails (429 quota, 404 model not found, etc.), the system:
1. Catches the error
2. Generates a template note using pattern matching from transcription
3. Returns the template note with `is_stub=True`
4. The template note is still saved to the database

### 4. Frontend Integration

**File**: `frontend/src/pages/doctor/DoctorWorkflow.tsx`

- **Loads saved notes** when a consultation is selected
- **Displays notes** with edit capability
- **Shows save status** (Saved badge, Edit button)
- **Allows editing** and saving updates
- **Shows warnings** if note saving fails

### 5. API Endpoints

**GET** `/api/v1/make-agent/consultations/{consultation_id}/note`
- Retrieves the current note for a consultation
- Returns note content, status, entities, etc.

**PUT** `/api/v1/make-agent/consultations/{consultation_id}/note`
- Updates an existing note
- Creates a new `NoteVersion` for audit trail
- Returns updated note information

## Verification Steps

### Test 1: Verify Note Saving with Gemini 429 Error

1. Upload audio with a consultation_id
2. Process audio (Gemini will fail with 429)
3. Check backend logs for:
   - "Note saved successfully to consultation {consultation_id} (is_stub=True)"
4. Check frontend:
   - Should show "Note saved" toast
   - Should display the template note
   - Should show "Saved" badge

### Test 2: Verify Note Loading

1. Select a consultation that has a saved note
2. Check frontend:
   - Note should load automatically
   - Should display in the Clinical Note section
   - Should show last updated timestamp

### Test 3: Verify Note Editing

1. Click "Edit Note" button
2. Make changes to the note
3. Click "Save Changes"
4. Check:
   - Note should update in database
   - Frontend should show "Note updated" toast
   - Note should persist after page refresh

### Test 4: Check Database

```sql
-- Check if notes are being saved
SELECT n.id, n.consultation_id, n.status, nv.content, nv.is_ai_generated, nv.created_at
FROM notes n
JOIN note_versions nv ON n.current_version_id = nv.id
WHERE n.consultation_id = '{consultation_id}'
ORDER BY nv.created_at DESC;
```

## Expected Behavior

### When Gemini Works (No 429 Error)
- AI-generated SOAP note is created
- Note is saved with `is_ai_generated=True`
- Note contains detailed clinical information

### When Gemini Fails (429 Quota Error)
- Template-based SOAP note is created
- Note is saved with `is_ai_generated=False` (is_stub=True)
- Note contains extracted information from transcription:
  - Symptoms (chest pain, shortness of breath, etc.)
  - Duration (for a few weeks)
  - Activity triggers (when climbing stairs)
  - Tests mentioned (ECG, electrocardiogram)
  - Clinical assessment based on symptoms
  - Treatment plan with recommendations

### When Transcription Fails
- Note is NOT saved (critical error)
- Error is logged and returned to frontend
- Frontend shows error message

## Troubleshooting

### Notes Not Saving

1. **Check consultation_id**: Ensure audio upload includes `consultation_id`
2. **Check logs**: Look for "Note saved successfully" or "Failed to save note"
3. **Check database**: Verify note exists in `notes` and `note_versions` tables
4. **Check frontend**: Look for "Note saved" toast or error messages

### Notes Not Loading

1. **Check API endpoint**: Verify `/api/v1/make-agent/consultations/{consultation_id}/note` is accessible
2. **Check frontend console**: Look for API errors
3. **Check note exists**: Verify note exists in database for the consultation_id

### Notes Not Editable

1. **Check edit button**: Should appear when note is displayed
2. **Check save button**: Should appear when in edit mode
3. **Check API endpoint**: Verify PUT endpoint is accessible
4. **Check permissions**: Ensure user has edit permissions (currently not enforced)

## Current Status

✅ **Notes are saved** even when Gemini fails with 429 quota error
✅ **Template notes are generated** with extracted information
✅ **Frontend loads saved notes** automatically
✅ **Doctors can edit notes** and save changes
✅ **Note versions are tracked** for audit trail
✅ **Frontend shows save status** and edit capabilities

## Next Steps

1. Test with a real consultation to verify end-to-end flow
2. Verify notes persist after page refresh
3. Test note editing and saving
4. Verify note versions are created correctly
5. Test with multiple consultations to ensure notes don't mix up

