# Scribe-Agent Service

The scribe-agent ingests dialogue transcripts and generates SOAP notes within a hospital boundary. It complements the manage-agent by structuring clinician-patient conversations and feeds outputs into the summarizer-agent.

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Health probe. |
| POST | `/scribe/transcript` | Store a dialogue transcript for an encounter. |
| GET | `/scribe/transcript` | List transcripts, optionally filtered by `encounter_id`. |
| POST | `/scribe/generate-soap` | Generate and persist a SOAP note (heuristic stub). |

### Example: Create Transcript

```http
POST /scribe/transcript
Content-Type: application/json

{
  "encounter_id": "6d4c3f04-9f6c-4b0e-b6f0-8cb5da291234",
  "transcript": "Doctor: How long have symptoms persisted?\nPatient: About three days.",
  "speaker_segments": [
    {"speaker": "doctor", "content": "How long have symptoms persisted?"},
    {"speaker": "patient", "content": "About three days."}
  ],
  "source": "scribe"
}
```

### Example: Generate SOAP Note

```http
POST /scribe/generate-soap
Content-Type: application/json

{
  "encounter_id": "6d4c3f04-9f6c-4b0e-b6f0-8cb5da291234",
  "transcript": "Patient complains of persistent cough lasting three days."
}
```

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r services/scribe_agent/requirements.txt

cp services/scribe_agent/env.example services/scribe_agent/.env

export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/scribe_agent"
poetry run alembic -c database/alembic.ini upgrade head

uvicorn services.scribe_agent.main:app --reload --port 8002
```

## Environment Variables

| Variable | Description |
| -------- | ----------- |
| `DATABASE_URL` | Async SQLAlchemy connection string. |
| `SCRIBE_AGENT_MODEL_VERSION` | Identifier for the SOAP generation model. |
| `SCRIBE_AGENT_CORS_ORIGINS` | Allowed origins for CORS. |

> Copy `services/scribe_agent/env.example` locally to `.env` because `.env`-prefixed files are not committed.

## Testing

```bash
TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/scribe_agent_test" \
poetry run pytest services/scribe_agent/tests
```

## Docker

```bash
docker build -f services/scribe_agent/Dockerfile -t scribe-agent:latest .
docker run --env-file services/scribe_agent/env.example -p 8002:8002 scribe-agent:latest
```

