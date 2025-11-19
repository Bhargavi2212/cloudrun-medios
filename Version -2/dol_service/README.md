# Data Orchestration Layer (DOL)

The DOL coordinates federated patient profile retrieval and federated learning metadata exchange among hospital instances. It aggregates local data, applies privacy filters, queries peer hospitals, and records audit events.

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/api/dol/health` | Health probe. |
| GET | `/api/dol/registry` | List hospitals registered with the orchestrator. |
| POST | `/api/dol/registry` | Register or heartbeat a hospital (requires shared secret). |
| POST | `/api/dol/patients/{patient_id}/snapshot` | Push sanitized demographics, summaries, and timeline events to the orchestrator. |
| GET | `/api/dol/patients/{patient_id}/profile` | Retrieve the cached federated profile for a patient. |
| POST | `/api/federated/patient` | Return a portable profile for a patient (requires shared secret). |
| POST | `/api/federated/timeline` | Provide local timeline fragment for peers. |
| POST | `/api/federated/model_update` | Accept federated learning model updates (acknowledgement stub). |

### Authentication
All federated endpoints expect an `Authorization: Bearer <shared_secret>` header. Optionally provide `X-Requester` to identify the peer for auditing.

### Patient Cache Workflow

1. Each hospital registers itself via `/api/dol/registry`.
2. After encounters or summarization, the hospital posts a snapshot to `/api/dol/patients/{patient_id}/snapshot`.
3. The orchestrator appends timeline events and stores sanitized demographics/summaries.
4. When a patient presents at a different hospital, that facility queries `/api/dol/patients/{patient_id}/profile` to obtain the full history without contacting the original hospital.

## Local Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r dol_service/requirements.txt

cp dol_service/env.example dol_service/.env

export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dol_service"
poetry run alembic -c database/alembic.ini upgrade head

uvicorn dol_service.main:app --reload --port 8004
```

## Environment Variables

| Variable | Description |
| -------- | ----------- |
| `DATABASE_URL` | Async SQLAlchemy connection string. |
| `DOL_HOSPITAL_ID` | Identifier for this hospital instance. |
| `DOL_SHARED_SECRET` | Shared secret for authenticating peers. |
| `DOL_CORS_ORIGINS` | Allowed origins for browser clients. |
| `DOL_PEERS` | JSON array of peer configs (`[{"name": "...", "base_url": "...", "api_key": "..."}]`). |

## Testing

```bash
TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/dol_service_test" \
poetry run pytest dol_service/tests
```

## Docker

```bash
docker build -f dol_service/Dockerfile -t dol-service:latest .
docker run --env-file dol_service/env.example -p 8004:8004 dol-service:latest
```
# Data Orchestration Layer (DOL)

This service coordinates portable patient profile requests between hospitals. It will expose:

- `/api/federated/patient` for profile retrieval
- `/api/federated/timeline` for timeline inspection
- `/api/federated/model_update` for federated learning passthrough
- `/api/dol/health` for monitoring

Implementation details, environment variables, and API documentation will be added during Phase 4 of the build plan.

