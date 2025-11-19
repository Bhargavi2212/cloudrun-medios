# Summarizer-Agent Service

The summarizer-agent produces longitudinal patient summaries for clinicians using locally stored encounters, transcripts, and SOAP notes. It aggregates highlights into append-only portable profiles that flow with the patient.

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Service health probe. |
| POST | `/summarizer/generate-summary` | Generate and persist a patient summary. |
| GET | `/summarizer/history/{patient_id}` | Retrieve summary history for a patient. |

### Example: Generate Summary

```http
POST /summarizer/generate-summary
Content-Type: application/json

{
  "patient_id": "8473f4d5-77f8-4f89-95a1-2d6a8a85f201",
  "encounter_ids": [
    "dd9cdf84-0c24-4975-94bc-bf7e719fa9f5",
    "9d6c29d2-f535-4f63-8f0c-e8e127980f53"
  ],
  "highlights": [
    "Encounter 1: Presented with migraine; treated with NSAIDs.",
    "Encounter 2: Follow-up with improved symptoms."
  ]
}
```

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r services/summarizer_agent/requirements.txt

cp services/summarizer_agent/env.example services/summarizer_agent/.env

export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/summarizer_agent"
poetry run alembic -c database/alembic.ini upgrade head

uvicorn services.summarizer_agent.main:app --reload --port 9003
```

## Environment Variables

| Variable | Description |
| -------- | ----------- |
| `DATABASE_URL` | Async SQLAlchemy connection string. |
| `SUMMARIZER_AGENT_MODEL_VERSION` | Active summarizer model identifier. |
| `SUMMARIZER_AGENT_CORS_ORIGINS` | Allowed origins for CORS. |

Copies of `.env` templates live at `services/summarizer_agent/env.example` for local setup.

## Testing

```bash
TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/summarizer_agent_test" \
poetry run pytest services/summarizer_agent/tests
```

## Docker

```bash
docker build -f services/summarizer_agent/Dockerfile -t summarizer-agent:latest .
docker run --env-file services/summarizer_agent/env.example -p 9003:9003 summarizer-agent:latest
```

