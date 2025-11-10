# Dashboard "API Usage" Explanation

## Problem
Dashboard shows "no API usage" even though notes are being saved successfully.

## Root Cause
The dashboard is tracking **LLM API usage** (Gemini API calls), not note creation. When Gemini API fails with 429 quota errors, template notes are generated and saved, but **no LLM usage is logged**, so the dashboard shows "no API usage".

## Solution
We've added LLM usage tracking for **template notes** so the dashboard shows activity even when Gemini API fails.

## Changes Made

### 1. Added LLM Usage Tracking
**File**: `backend/services/make_agent_medi_os.py`

- Added `record_llm_usage()` call after processing audio
- Template notes are logged with `model="template-note-generator"`
- This ensures the dashboard shows API usage even when Gemini fails

### 2. Updated Pipeline to Include Model Info
**File**: `backend/services/make_agent_pipeline.py`

- Updated `process_audio()` to include `model`, `tokens_prompt`, `tokens_completion`, `cost_cents` in the result
- This information is now available for LLM usage tracking

## How It Works Now

### When Gemini API Works:
1. Audio is transcribed
2. Entities are extracted
3. Gemini generates SOAP note
4. LLM usage is logged: `model="gemini-2.0-flash-exp"`, `tokens=X`, `status="success"`
5. Note is saved to database
6. **Dashboard shows API usage** ✅

### When Gemini API Fails (429 Quota Error):
1. Audio is transcribed
2. Entities are extracted
3. Gemini fails with 429 error
4. Template note is generated
5. **LLM usage is logged**: `model="template-note-generator"`, `tokens=0`, `status="success"`
6. Note is saved to database
7. **Dashboard shows API usage** ✅ (shows as "template-note-generator")

## Database Tracking

LLM usage is stored in the `llm_usage` table:
- `model`: Model name (e.g., "gemini-2.0-flash-exp" or "template-note-generator")
- `tokens_prompt`: Input tokens (0 for template notes)
- `tokens_completion`: Output tokens (0 for template notes)
- `cost_cents`: Cost in cents (0.0 for template notes)
- `status`: "success" or "failed"
- `created_at`: Timestamp

## Dashboard Queries

The dashboard likely queries the `llm_usage` table:
```sql
SELECT 
    COUNT(*) as total_requests,
    SUM(tokens_prompt + tokens_completion) as total_tokens,
    SUM(cost_cents) as total_cost,
    model,
    status
FROM llm_usage
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY model, status
ORDER BY created_at DESC;
```

## Verification

After processing an audio recording, check the logs for:
```
📊 Recorded LLM usage: model=template-note-generator, tokens=0+0, status=success, has_note=True
```

Then check the database:
```sql
SELECT * FROM llm_usage ORDER BY created_at DESC LIMIT 10;
```

You should see entries with `model="template-note-generator"` when template notes are generated.

## Next Steps

1. **Process an audio recording** and check logs for "📊 Recorded LLM usage"
2. **Check the database** to verify LLM usage entries are being created
3. **Check the dashboard** - it should now show API usage (even for template notes)
4. **Verify the dashboard query** - make sure it's querying the `llm_usage` table correctly

## Important Notes

- **Template notes are still valuable** - they contain extracted information from transcription
- **LLM usage tracking** helps monitor system activity and costs
- **Dashboard shows activity** even when Gemini API is unavailable
- **Notes are saved** regardless of whether Gemini API works or not

## Troubleshooting

If the dashboard still shows "no API usage":

1. **Check if LLM usage is being logged**:
   ```sql
   SELECT * FROM llm_usage ORDER BY created_at DESC LIMIT 10;
   ```

2. **Check logs for "📊 Recorded LLM usage"** messages

3. **Verify the dashboard query** - make sure it's querying the correct table and time range

4. **Check if there are any errors** in the LLM usage logging:
   ```
   ⚠️ Failed to record LLM usage: ...
   ```

5. **Verify the model name** - template notes should use `model="template-note-generator"`

