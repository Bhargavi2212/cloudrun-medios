## Medi OS Version -2 Planning

### Purpose
- Establish a clean-slate implementation of Medi OS inside `Version -2`, starting with database foundations.
- Capture architectural decisions up front so backend, frontend, and deployment align with project rules.
- Provide a sequenced roadmap to avoid shortcuts (e.g., fake print statements) and ensure production readiness.

### Guiding Principles
- Follow `.cursorrules` for architecture, naming, and quality standards.
- Keep services decoupled; communicate via documented REST APIs.
- Treat environment parity seriously: anything we build must ship without ad-hoc tweaks.
- Prefer explicit types, thorough docstrings, and async I/O everywhere.

### Technology Stack
- **Database:** PostgreSQL 15 (Cloud SQL compatible); SQLAlchemy + Alembic for migrations.
- **Backend Services:** Three FastAPI services (`manage-agent`, `scribe-agent`, `summarizer-agent`) with Python 3.11, Pydantic, and asyncio.
- **Frontend:** React 18 + TypeScript + Vite + MUI; Axios for API integration.
- **Infrastructure:** Dockerized microservices targeting Google Cloud Run; Terraform module to follow after core build.

### Developer Environment
- Follow `.tool-versions` (Python 3.11.6, Node 18.18.2) or `.nvmrc` for local runtime alignment.
- Install Poetry (`pipx install poetry`) and bootstrap shared tooling with `poetry install` from the `Version -2/` directory.
- Enable quality gates: `pre-commit install`, then run `poetry run ruff check .`, `poetry run black --check .`, `poetry run mypy .`, and `poetry run pytest --collect-only` before commits.
- Frontend dependencies will be initialized in Phase 6 using Vite (`npm create vite@latest` or `pnpm`), tracked under `apps/frontend/`.
- Integration tests for database modules expect `TEST_DATABASE_URL` to point to a disposable PostgreSQL instance (e.g., `postgresql+asyncpg://user:pass@localhost:5432/medi_os_test`).

### Core Domain Model (Initial Draft)
- **patients**
  - `id` (UUID PK)
  - `mrn` (string, unique)
  - `first_name`, `last_name`, `dob`, `sex`, `contact_info`
  - `created_at`, `updated_at`
- **encounters**
  - `id` (UUID PK)
  - `patient_id` (FK → patients.id)
  - `arrival_ts`, `disposition`, `location`
  - `acuity_level` (nullable until triage complete)
  - `created_at`, `updated_at`
- **triage_observations**
  - `id` (UUID PK)
  - `encounter_id` (FK → encounters.id)
  - `vitals` (JSONB storing structured vitals with ranges)
  - `chief_complaint`, `notes`
  - `triage_score`, `triage_model_version`
  - `created_at`
- **dialogue_transcripts**
  - `id` (UUID PK)
  - `encounter_id` (FK)
  - `transcript` (TEXT)
  - `speaker_segments` (JSONB)
  - `source` (`scribe`, `import`, etc.)
  - `created_at`
- **soap_notes**
  - `id` (UUID PK)
  - `encounter_id` (FK)
  - `subjective`, `objective`, `assessment`, `plan`
  - `model_version`, `confidence_score`
  - `created_at`, `updated_at`
- **summaries**
  - `id` (UUID PK)
  - `patient_id` (FK)
  - `encounter_ids` (JSONB list for provenance)
  - `summary_text`, `model_version`, `confidence_score`
  - `created_at`
- **audit_logs**
  - `id` (UUID PK)
  - `entity_type`, `entity_id`, `action`, `performed_by`, `metadata`
  - `created_at`

### Database Plan
- Use Alembic migrations from day one; no manual schema drift.
- Seed scripts limited to non-sensitive reference data (e.g., triage codes). All real datasets stay in `/data`.
- Enforce optimistic concurrency with `updated_at` and version counters where needed.
- Add row-level security policies once baseline CRUD works; align with service auth design.

### Service Responsibilities
- **manage-agent**
  - Owns triage business logic, vitals ingestion, wait-time estimation.
  - CRUD for `patients`, `encounters`, `triage_observations`.
  - Exposes `/manage/classify`, `/manage/wait-time`, `/manage/encounter`.
- **scribe-agent**
  - Handles dialogue ingestion and SOAP generation.
  - Writes to `dialogue_transcripts`, `soap_notes`.
  - Exposes `/scribe/generate-soap`, `/scribe/transcript`.
- **summarizer-agent**
  - Reads consolidated data to produce longitudinal summaries.
  - Maintains `summaries` table with provenance references.
  - Exposes `/summarizer/generate-summary`, `/summarizer/history`.

### Cross-Service Integration
- Standardize request/response Pydantic models in each service under `dto/`.
- Use async `httpx` clients with retry + timeout for inter-service calls.
- Eventual addition: lightweight event log (e.g., Pub/Sub) for decoupled notifications; not in initial scope.

### Frontend Alignment
- Feature folders: `triage`, `scribe`, `summary`.
- Shared TypeScript types generated from OpenAPI specs (via `backend/scripts/generate_sdk.py` equivalent in Version -2).
- Material UI theme configured once; adhere to accessibility guidelines.

### Security & Compliance Checklist
- No PHI in logs; leverage structured logging with redaction middleware.
- JWT-based auth gateway (TBD) with service-level API keys for now.
- Encrypt data-in-transit (HTTPS) and enable at-rest encryption through managed Postgres.

### Implementation Roadmap
1. **Database Bootstrap**
   - Configure SQLAlchemy models & Alembic migration scaffolding.
   - Create base `database` package with session management and async engine.
   - Write initial migration covering core tables above.
2. **Shared Utilities**
   - Logging, settings management, DTO base classes, error schemas.
   - Define Pydantic config models for environment variables.
3. **Manage-Agent MVP**
   - Implement FastAPI skeleton, health endpoint, triage classification stub with TODO for real model loading.
   - Integrate database CRUD for patient + encounter lifecycle.
   - Add pytest coverage for repositories and handlers.
4. **Scribe-Agent MVP**
   - Dialogue ingestion endpoint.
   - SOAP generation pipeline with placeholder integration (real model to follow).
5. **Summarizer-Agent MVP**
   - Summary generation endpoint referencing historical data.
6. **Frontend Scaffold**
   - Vite project under `Version -2/apps/frontend` with routing and API client bootstrap.
7. **CI/CD Foundations**
   - Dockerfiles per service, docker-compose for local orchestration, GitHub Actions (or Cloud Build) skeleton.
8. **Model Integration**
   - Replace stubs with actual model loaders and inference pipelines.
9. **Observability**
   - Structured logging, tracing hooks, and health metrics.

### Repository Layout (Bootstrap)
- `.github/workflows/` — CI pipelines (linting, typing, pytest collection).
- `.config/` — shared configuration templates.
- `shared/` — cross-service helpers (config loading, logging, exceptions).
- `database/`, `federation/`, `dol_service/`, `services/` — backend packages and services.
- `apps/frontend/` — React application scaffold (Phase 6).
- `docs/`, `scripts/`, `tests/` — documentation, automation scripts, and shared test utilities.

### Quickstart Commands
- Seed demo data: `poetry run python scripts/bootstrap_hospital_instance.py`
- Launch demo stack: `docker compose -f docker-compose.demo.yml up --build`
- Reset demo data: `./scripts/reset_demo_data.sh`
- Submit sample federated round: `./scripts/run_federated_round.sh triage`

### Next Actions
- Confirm database schema requirements with stakeholders (e.g., additional fields for vitals provenance or audit).
- Initialize Python project layout under `Version -2/services/manage_agent` with Poetry/PDM or pip-tools (decision pending).
- Draft Alembic migration `versions/<timestamp>_initial_schema.py` reflecting tables above.

---
All subsequent implementation steps must update this plan if assumptions change. No fake output or doctored tests—every deliverable must function in production-equivalent environments.

