# Manage-Agent Service

The manage-agent orchestrates patient intake, triage classification, and encounter lifecycle management for a single hospital instance. It backs the federated Medi OS workflow by persisting patients, encounters, and triage observations in the local PostgreSQL database.

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/health` | Health probe returning service status. |
| POST | `/manage/patients` | Create a patient record. |
| GET | `/manage/patients` | List patients with pagination. |
| GET | `/manage/patients/{patient_id}` | Retrieve patient details. |
| PUT | `/manage/patients/{patient_id}` | Update patient fields. |
| DELETE | `/manage/patients/{patient_id}` | Delete a patient. |
| POST | `/manage/classify` | Generate a triage acuity score (stubbed heuristic). |
| POST | `/manage/patients/{patient_id}/check-in` | Fetch portable profile via the DOL automatically. |

### Example: Create Patient

```http
POST /manage/patients
Content-Type: application/json

{
  "mrn": "MED-100001",
  "first_name": "Alex",
  "last_name": "Kim",
  "sex": "other"
}
```

### Example: Triage Classification

```http
POST /manage/classify
Content-Type: application/json

{
  "hr": 110,
  "rr": 22,
  "sbp": 118,
  "dbp": 70,
  "temp_c": 37.5,
  "spo2": 95,
  "pain": 3
}
```

### Example: Patient Check-In

```http
POST /manage/patients/6d4c3f04-9f6c-4b0e-b6f0-8cb5da291234/check-in
Authorization: Bearer <internal service-to-service credential>
```


## Local Development

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r services/manage_agent/requirements.txt

# Copy and adjust environment variables
cp services/manage_agent/env.example services/manage_agent/.env

# Run database migrations (from repository root)
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/manage_agent"
poetry run alembic -c database/alembic.ini upgrade head

# Launch the API
uvicorn services.manage_agent.main:app --reload --port 8001
```

## Environment Variables

| Variable | Description |
| -------- | ----------- |
| `DATABASE_URL` | Async SQLAlchemy connection string for local Postgres. |
| `MANAGE_AGENT_MODEL_VERSION` | Identifier for the triage model in use. |
| `MANAGE_AGENT_CORS_ORIGINS` | Comma-separated list of allowed origins (e.g., frontend host). |
| `MANAGE_AGENT_DOL_URL` | Base URL for contacting the DOL service. |
| `MANAGE_AGENT_DOL_SECRET` | Shared secret used when calling the DOL. |
| `MANAGE_AGENT_HOSPITAL_ID` | Identifier sent to the DOL for audit logging. |

> **Note:** Some tooling blocks committing `.env`-prefixed files. Use `services/manage_agent/env.example` as the canonical template and copy it locally to `.env`.

## Testing

```
TEST_DATABASE_URL="postgresql+asyncpg://user:pass@localhost:5432/manage_agent_test" \
poetry run pytest services/manage_agent/tests
```

## Docker

```
docker build -f services/manage_agent/Dockerfile -t manage-agent:latest .
docker run --env-file services/manage_agent/env.example -p 8001:8001 manage-agent:latest
```

