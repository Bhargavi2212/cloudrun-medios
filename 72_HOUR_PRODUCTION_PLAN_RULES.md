# Medi OS 72-Hour Production Plan - Cursor Implementation Rules

**Last Updated**: 2025-11-10  
**Status**: In Progress - Phase 1-2 mostly complete, Phase 3-6 in progress

## Current Progress Summary

### ✅ Completed (~70% Overall)
- **Phase 1**: Database models, JWT auth, storage abstraction (80% complete)
- **Phase 2**: Triage, summarizer, queue engine, basic scribe (70% complete)
- **Phase 3**: Basic dashboards, auth UI (60% complete)

### ⚠️ In Progress
- **Phase 1**: Settings/secrets management, Cloud Secret Manager integration
- **Phase 2**: Scribe async processing, API orchestration improvements
- **Phase 3**: Advanced UI features, note approval workflows
- **Phase 4**: Test coverage expansion, load testing
- **Phase 5**: GCP infrastructure, CI/CD pipeline
- **Phase 6**: Documentation completion

## Implementation Rules for Cursor

### CRITICAL: Follow Phase Order
1. **NEVER skip phases** - Complete Phase 1 before Phase 2, etc.
2. **Check existing code first** - Many components already exist
3. **Enhance, don't recreate** - Build upon existing implementations
4. **Update progress** - Mark completed items in this file

### Phase 1 Rules: Foundation (12-15h)

#### db-init
- **Location**: `backend/database/models.py`, `backend/database/migrations/`
- **Status**: ✅ Models exist, migrations exist
- **Action**: Verify all models from plan exist, add missing ones
- **Rules**:
  - All models must inherit: `Base`, `TimestampMixin`, `UUIDPrimaryKeyMixin`
  - Use `SoftDeleteMixin` for soft-deletable entities
  - Create Alembic migrations: `alembic revision --autogenerate -m "description"`
  - Never modify existing migrations - create new ones
  - Seed scripts in `backend/database/seeds.py`
  - Connection pooling: `pool_size=10, max_overflow=20` in `backend/database/session.py`

#### auth-core
- **Location**: `backend/security/jwt.py`, `backend/api/v1/auth.py`
- **Status**: ✅ JWT exists, refresh tokens exist
- **Action**: Add refresh token rotation, email flows
- **Rules**:
  - JWT in `backend/security/jwt.py` (already exists)
  - Password hashing in `backend/security/password.py` (already exists)
  - Role guard in `backend/security/permissions.py` (already exists)
  - Add refresh token rotation logic
  - Create `backend/services/email_service.py` for email verification/reset
  - Audit logging: Use existing `AuditLog` model

#### storage-config
- **Location**: `backend/services/storage.py`
- **Status**: ✅ Local + GCS abstraction exists
- **Action**: Add retention/backups policy hooks
- **Rules**:
  - Storage abstraction already supports local + GCS
  - Add `retention_policy` field to `FileAsset` model
  - Implement retention cleanup job in `backend/services/job_queue.py`
  - Signed URLs: Enhance existing `generate_signed_url()` method

#### settings-secrets
- **Location**: `backend/services/config.py`
- **Status**: ⚠️ Basic config exists, needs env tiers
- **Action**: Add dev/staging/prod tiers, Cloud Secret Manager
- **Rules**:
  - Expand `Settings` class with `environment` field
  - Add `get_secret(secret_name: str)` method using Cloud Secret Manager
  - Rate limiting: Use existing `backend/services/rate_limit.py`
  - Feature flags: Use existing `backend/services/feature_flags.py`
  - Never hardcode secrets - always use `os.getenv()` or `get_secret()`

### Phase 2 Rules: Backend Services (10-12h)

#### scribe-upgrade
- **Location**: `backend/services/make_agent_medi_os.py`, `backend/api/v1/make_agent.py`
- **Status**: ⚠️ Basic pipeline exists, needs async processing
- **Action**: Add background jobs, streaming, note approval
- **Rules**:
  - Use existing `backend/services/job_queue.py` for async processing
  - Add `POST /api/v1/make-agent/process-async` endpoint
  - Streaming: Use `FastAPI.StreamingResponse` for real-time transcription
  - Note versioning: Already exists in `NoteVersion` model
  - Note approval: Add `status` field to `Note` model, create approval workflow
  - Telemetry: Use existing `backend/services/telemetry.py`

#### triage-wrapper
- **Location**: `backend/api/v1/triage.py`, `backend/services/triage_service.py`
- **Status**: ✅ Endpoints exist
- **Action**: Add SHAP explainability, SLA logging
- **Rules**:
  - Endpoints already exist in `backend/api/v1/triage.py`
  - Add `explain_prediction()` method using SHAP library
  - SLA logging: Use `backend/services/telemetry.py`
  - Add `/api/v1/triage/explain` endpoint

#### summarizer-wrapper
- **Location**: `backend/services/summarizer_service.py`, `backend/api/v1/summarizer.py`
- **Status**: ✅ Service exists
- **Action**: Add caching, cost/latency tracking
- **Rules**:
  - Caching: Use Postgres materialized views or Redis
  - Cost tracking: Use `LLMUsage` model in `backend/database/models.py`
  - Latency tracking: Use `backend/services/telemetry.py`
  - Add `/api/v1/historian` endpoints if needed

#### queue-engine
- **Location**: `backend/services/manage_agent_state_machine.py`, `backend/api/v1/queue.py`
- **Status**: ✅ State machine exists
- **Action**: Add Server-Sent Events, WebSocket fallback
- **Rules**:
  - State machine already exists
  - Wait-time estimation: Use existing `backend/services/wait_time_estimator.py`
  - Server-Sent Events: Add `GET /api/v1/queue/stream` endpoint
  - WebSocket: Use Postgres `LISTEN/NOTIFY` for real-time updates
  - Fallback: Use polling if WebSocket unavailable

#### api-orchestration
- **Location**: `backend/api/v1/__init__.py`, `backend/api/error_handlers.py`
- **Status**: ⚠️ Routers exist, needs unification
- **Action**: Unified error handling, OpenAPI docs, correlation-ID
- **Rules**:
  - Routers already registered in `backend/api/v1/__init__.py`
  - Error handling: Enhance `backend/api/error_handlers.py`
  - OpenAPI: Customize in `backend/main.py` with `app.openapi()`
  - Correlation-ID: Add middleware in `backend/services/middleware.py`
  - SDK generation: Create `backend/scripts/generate_sdk.py`

### Phase 3 Rules: Frontend Completion (8-10h)

#### auth-ui
- **Location**: `frontend/src/pages/LoginPage.tsx`, `frontend/src/stores/authStore.ts`
- **Status**: ⚠️ Login exists, needs register/reset
- **Action**: Add register, reset password, session timeout
- **Rules**:
  - Login: Already exists
  - Register: Create `frontend/src/pages/RegisterPage.tsx`
  - Reset: Create `frontend/src/pages/ResetPasswordPage.tsx`
  - Session timeout: Add to `frontend/src/stores/authStore.ts`
  - Account settings: Create `frontend/src/pages/AccountSettings.tsx`

#### dashboards
- **Location**: `frontend/src/pages/receptionist/`, `frontend/src/pages/nurse/`, `frontend/src/pages/doctor/`
- **Status**: ✅ Dashboards exist
- **Action**: Add live data, charts, filters, responsive design
- **Rules**:
  - Use React Query for live data: `useQuery` with `refetchInterval`
  - Charts: Use `recharts` library
  - Filters: Add filter components using Radix UI
  - Responsive: Use Tailwind CSS breakpoints (`sm:`, `md:`, `lg:`)

#### ai-interfaces
- **Location**: `frontend/src/pages/doctor/DoctorWorkflow.tsx`
- **Status**: ⚠️ Basic interfaces exist
- **Action**: Add status indicators, history views, download/export
- **Rules**:
  - Status indicators: Use Badge components with loading/error states
  - History views: Create `frontend/src/pages/HistoryPage.tsx`
  - Download: Add download buttons with `window.open()` or `fetch().blob()`
  - Export: Add export to PDF/CSV functionality

#### advanced-ui
- **Location**: `frontend/src/pages/`
- **Status**: ⚠️ Patient search exists
- **Action**: Add consultation history, note approval workflows
- **Rules**:
  - Patient search: Enhance existing search with filters
  - Consultation history: Create `frontend/src/pages/ConsultationHistory.tsx`
  - Note approval: Add approval workflow in `frontend/src/pages/doctor/DoctorWorkflow.tsx`
  - Accessibility: Use ARIA labels, keyboard navigation
  - Dark mode: Add theme toggle using Tailwind dark mode

### Phase 4 Rules: Testing & QA (8-10h)

#### backend-tests
- **Location**: `backend/tests/`
- **Status**: ⚠️ Basic tests exist
- **Action**: Achieve ≥80% coverage, add integration tests
- **Rules**:
  - Unit tests: Use pytest, target ≥80% coverage
  - Integration tests: Test full workflows (check-in → triage → consultation)
  - Concurrent access: Use `pytest-xdist` for parallel testing
  - Mock external services: Use `pytest-mock` or `unittest.mock`
  - Coverage: Run `pytest --cov=backend --cov-report=html`

#### frontend-tests
- **Location**: `frontend/src/__tests__/`
- **Status**: ❌ Need to create
- **Action**: Add Jest unit tests, Playwright E2E tests
- **Rules**:
  - Unit tests: Use Vitest (already configured)
  - E2E tests: Use Playwright (preferred) or Cypress
  - Core role flows: Test receptionist → nurse → doctor workflows
  - Setup: `npm install -D @playwright/test`

#### load-tests
- **Location**: `tests/load/`
- **Status**: ❌ Need to create
- **Action**: Create k6 or locust scripts
- **Rules**:
  - Use k6 (preferred) or locust
  - Test triage/scribe/summarizer queues
  - Document performance tuning recommendations
  - Create `tests/load/triage.js`, `scribe.js`, `summarizer.js`

### Phase 5 Rules: Deployment & Ops (6-8h)

#### containerization
- **Location**: `backend/Dockerfile`, `docker-compose.yml`
- **Status**: ⚠️ Dockerfile exists
- **Action**: Enhance Dockerfiles, create docker-compose
- **Rules**:
  - Multi-stage Dockerfiles: Enhance existing `backend/Dockerfile`
  - docker-compose: Create `docker-compose.yml` with Postgres, nginx
  - Health checks: Add `/health` endpoints (already exist)
  - nginx: Create `nginx.conf` for reverse proxy

#### gcp-infra
- **Location**: `infra/`, `scripts/`
- **Status**: ❌ Need to create
- **Action**: Create Cloud Run scripts, Terraform modules
- **Rules**:
  - Cloud Run: Create `scripts/deploy-cloud-run.sh`
  - Cloud SQL: Create `scripts/setup-cloud-sql.sh`
  - Cloud Storage: Configure in `backend/services/storage.py`
  - Secret Manager: Integrate in `backend/services/config.py`
  - Terraform: Create `infra/terraform/` with modules

#### cicd
- **Location**: `.github/workflows/`
- **Status**: ❌ Need to create
- **Action**: Create GitHub Actions pipeline
- **Rules**:
  - Create `.github/workflows/ci-cd.yml`
  - Lint: Run black, isort, flake8, mypy
  - Tests: Run pytest with coverage
  - Build: Build Docker images
  - Deploy: Deploy to Cloud Run
  - Smoke tests: Add post-deployment health checks

#### monitoring
- **Location**: `backend/services/telemetry.py`
- **Status**: ⚠️ Basic telemetry exists
- **Action**: Add Prometheus metrics, Cloud Monitoring
- **Rules**:
  - Prometheus: Use `prometheus-fastapi-instrumentator`
  - Log structure: Use structured logging (JSON)
  - Cloud Monitoring: Create dashboards in GCP Console
  - Alerting: Set up alerting policies

### Phase 6 Rules: Documentation & Runbooks (4-6h)

#### dev-docs
- **Location**: `docs/`
- **Status**: ⚠️ Basic README exists
- **Action**: Create architecture, API reference, deployment guides
- **Rules**:
  - Architecture: Create `docs/architecture.md`
  - Setup: Create `docs/setup.md`
  - API reference: Auto-generate from OpenAPI, enhance manually
  - Deployment: Create `docs/deployment.md`
  - Troubleshooting: Create `docs/troubleshooting.md`

#### user-docs
- **Location**: `docs/user-guides/`
- **Status**: ❌ Need to create
- **Action**: Create role workflows, AI feature guides
- **Rules**:
  - Role workflows: Create `docs/user-guides/receptionist.md`, `nurse.md`, `doctor.md`, `admin.md`
  - AI features: Create `docs/user-guides/ai-scribe.md`, `ai-summarizer.md`, `triage.md`
  - FAQs: Create `docs/faq.md`
  - Screenshots: Add to user guides

#### ops-runbooks
- **Location**: `docs/runbooks/`
- **Status**: ❌ Need to create
- **Action**: Create model updates, database maintenance, incident response
- **Rules**:
  - Model updates: Create `docs/runbooks/model-updates.md`
  - Database maintenance: Create `docs/runbooks/database-maintenance.md`
  - Incident response: Create `docs/runbooks/incident-response.md`

## Code Quality Standards

### Python
- Format: `black`, `isort`
- Lint: `flake8`, `mypy`
- Tests: `pytest` with ≥80% coverage
- Docstrings: Google style
- Type hints: Required for all functions

### TypeScript
- Format: `prettier`
- Lint: `ESLint`
- Tests: `Vitest` for unit, `Playwright` for E2E
- JSDoc: Required for all functions
- Types: No `any` types allowed

### Git
- Commits: Conventional Commits format
- Branches: Feature branches for all work
- PRs: Required for all changes
- Never commit: Secrets, large files, `.env` files

## Security Requirements

1. **Never hardcode secrets** - Use environment variables
2. **Rate limiting** - On all public endpoints
3. **HTTPS only** - In production
4. **CORS** - Configure properly
5. **Input sanitization** - All user inputs
6. **Parameterized queries** - SQLAlchemy handles this
7. **Audit logging** - For sensitive operations

## Performance Requirements

1. **API response times** - < 2 seconds
2. **Database queries** - Use indexes, avoid N+1
3. **Caching** - Redis or in-memory for frequent data
4. **Background jobs** - For long-running tasks
5. **Connection pooling** - Configure properly

## Error Handling

1. **Structured responses** - `{"success": bool, "error": str, "data": any}`
2. **Logging** - All errors with context (request ID, user ID)
3. **HTTP status codes** - Appropriate codes
4. **User-friendly messages** - Clear error messages
5. **Retry logic** - For external API calls

## File Structure

```
backend/
├── database/          # Models, migrations, CRUD
├── api/v1/            # API endpoints
├── services/          # Business logic
├── security/          # Auth, JWT, permissions
└── tests/             # All tests

frontend/
├── src/
│   ├── pages/        # Page components
│   ├── components/   # Reusable components
│   ├── services/     # API clients
│   ├── stores/       # Zustand stores
│   └── hooks/       # Custom hooks
└── tests/           # Frontend tests

docs/                # Documentation
infra/               # Infrastructure as code
scripts/             # Deployment scripts
```

## Success Criteria

### Phase 1 Complete When:
- [ ] All database models defined and migrated
- [ ] JWT auth working with refresh tokens
- [ ] Storage abstraction supports local and GCS
- [ ] Environment configuration supports dev/staging/prod
- [ ] All secrets loaded from environment variables

### Phase 2 Complete When:
- [ ] All backend services have async background processing
- [ ] All endpoints have proper error handling and logging
- [ ] OpenAPI documentation is complete
- [ ] Telemetry is tracking all operations

### Phase 3 Complete When:
- [ ] All user roles can complete their workflows
- [ ] All dashboards show live data
- [ ] AI interfaces have status indicators and history
- [ ] Frontend is responsive and accessible

### Phase 4 Complete When:
- [ ] Backend test coverage ≥80%
- [ ] Frontend has unit and E2E tests
- [ ] Load tests are passing
- [ ] All tests are in CI/CD pipeline

### Phase 5 Complete When:
- [ ] Application is containerized and deployable
- [ ] GCP infrastructure is provisioned
- [ ] CI/CD pipeline is working
- [ ] Monitoring and alerting are configured

### Phase 6 Complete When:
- [ ] All documentation is written
- [ ] User guides are complete
- [ ] Runbooks are available
- [ ] API reference is documented

## Implementation Priority

1. **Critical Path**: Phase 1 → Phase 2 → Phase 3 (core functionality)
2. **Quality Assurance**: Phase 4 (testing) should run parallel to Phase 2-3
3. **Production Readiness**: Phase 5 (deployment) should start after Phase 2
4. **Documentation**: Phase 6 can be done incrementally throughout

## Notes for Cursor

- **Always check existing code first** - Many components already exist
- **Follow existing patterns** - Don't reinvent the wheel
- **Update this file** - Mark progress as you complete tasks
- **Ask for clarification** - If requirements are ambiguous
- **Prioritize working code** - Over perfect code
- **Test incrementally** - Don't wait until the end
- **Document as you go** - Don't leave it for later

