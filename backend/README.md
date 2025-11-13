# MediOS AI Scribe v2 (Hackathon Build)

Hardened implementation of the clinical scribe pipeline designed for Cloud Run deployment. The stack uses Whisper for speech-to-text, lightweight keyword extraction, and Gemini 1.5 Pro for SOAP note generation.

## Quick Start

```bash
cp env.example .env  # set GEMINI_API_KEY (and optional HF_TOKEN)
python -m venv .venv
. .venv/bin/activate  # use `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Demo Credentials

Cloud Run deployments seed four accounts automatically so reviewers can exercise every workflow without additional setup:

| Role | Email | Password | Highlights |
| --- | --- | --- | --- |
| Receptionist | `receptionist@medios.ai` | `Password123!` | Patient registration, queue management |
| Nurse | `nurse@medios.ai` | `Password123!` | Vitals capture, AI triage, document upload |
| Doctor | `doctor@medios.ai` | `Password123!` | Consultation workspace, AI Scribe, note approval |
| Admin | `admin@medios.ai` | `Password123!` | User management, analytics, system configuration |

Use these for quick demos, or register a new account via the public `/auth/register` endpoint selecting the appropriate role.

## Environment Variables

| Key                | Description                                           | Default              |
| ------------------ | ----------------------------------------------------- | -------------------- |
| `GEMINI_API_KEY`   | Google Generative AI key (required for live notes)    | _None_               |
| `HF_TOKEN`         | Optional HuggingFace token for authenticated pulls    | _None_               |
| `MODEL_SIZE`       | Whisper model alias (`tiny`, `base`, `small`, ...)    | `base`               |
| `DATA_PATH`        | Directory for uploaded audio                          | `./uploads`          |
| `WHISPER_CACHE_DIR`| Directory for cached Whisper weights                  | `./models`           |
| `GEMINI_MODEL`     | Gemini model id                                       | `models/gemini-1.5-pro` |
| `GEMINI_TEMPERATURE` | Sampling temperature                                | `0.2`                |
| `GEMINI_MAX_TOKENS` | Maximum output tokens                                | `1000`               |

## API Overview

| Endpoint                | Method | Description                              |
| ----------------------- | ------ | ---------------------------------------- |
| `/api/v1/make-agent/health`       | GET    | Health probe                            |
| `/api/v1/make-agent/upload`       | POST   | Upload audio, returns `audio_id`        |
| `/api/v1/make-agent/transcribe`   | POST   | Transcribe previously uploaded audio    |
| `/api/v1/make-agent/extract_entities` | POST | Keyword-based entity extraction         |
| `/api/v1/make-agent/generate_note`| POST   | Generate SOAP note via Gemini           |
| `/api/v1/make-agent/process`      | POST   | Run full pipeline (transcribe→note)     |
| `/api/v1/make-agent/update_note`  | PUT    | Stub endpoint for manual edits          |

All responses are wrapped in the `StandardResponse` schema:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "is_stub": false,
  "warning": null
}
```

## Docker & Cloud Run

```bash
docker build -t scribe .
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  scribe
```

Cloud Run deploy:

```bash
gcloud run deploy scribe \
  --source . \
  --region us-central1 \
  --set-secrets="GEMINI_API_KEY=gemini-key:latest,HF_TOKEN=hf-token:latest" \
  --memory 4Gi \
  --timeout 120s \
  --allow-unauthenticated
```

## Tests

Unit tests (with mocked models) live under `backend/tests/`. Run them with:

```bash
pytest
```

## Limitations

- Keyword entity extraction is intentionally lightweight for the hackathon; swap in a more robust NER model post-event.
- `/update_note` is a stub (no persistence) to keep the demo simple.
- Ensure `ffmpeg` is installed locally or use the Docker image for consistent behaviour.

