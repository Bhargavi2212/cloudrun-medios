# GitHub Actions Workflows

This directory contains CI/CD workflows for the Medi OS Version 2 project.

## Workflows

### Version 2 (Active)
#### `version-2-test.yml` - Test Suite
Runs backend and frontend tests for both HospitalA and HospitalB on every push and pull request.

**Features:**
- Tests HospitalA backend with Poetry
- Tests HospitalB backend with Poetry
- Tests frontend (from HospitalA directory)

**Note:** Tests are set to `continue-on-error: true` to allow workflow completion even with test failures.

#### `version-2-ci-cd.yml` - Full CI/CD Pipeline
Complete CI/CD pipeline for Version 2 microservices architecture.

**Stages:**
1. **Hospital A Backend** - Linting (Ruff, Black), type checking (MyPy), and tests
2. **Hospital B Backend** - Linting (Ruff, Black), type checking (MyPy), and tests
3. **Frontend** - Linting (ESLint), type checking (TypeScript), and tests
4. **Build Docker Images** - Builds all microservices:
   - manage-agent (port 8001/9001)
   - scribe-agent (port 8002/9002)
   - summarizer-agent (port 8003/9003)
   - dol-service (port 8004/9004)
   - federation-aggregator (port 8010/9010)
   - frontend
5. **Deploy to Cloud Run** - Deploys all services to GCP Cloud Run (only on main branch)

**Services Deployed:**
- Each hospital gets its own set of services
- Both hospitals deploy to `us-central1` region (same region = lower cost, lower latency)
- All services built with Docker and pushed to Google Container Registry
- **Note:** You can change the region in the workflow file if needed (e.g., us-east1, europe-west1, asia-east1)

**Required Secrets:**
- `GCP_SA_KEY` - Google Cloud Service Account JSON key
- `GCP_PROJECT_ID` - Google Cloud Project ID

**Triggers:**
- Push to `main` or `develop`: Runs full pipeline (lint, test, build, deploy)
- Pull Request: Runs lint and test only (no build/deploy)
- Manual dispatch: Can be triggered manually

**Note:** Tests and deployments are set to `continue-on-error: true` to allow workflow completion even with failures. This helps identify all issues at once rather than stopping on the first failure.

## Legacy Workflows (Removed)

The old workflows (`ci-cd.yml` and `test.yml`) for the legacy `backend/` and `frontend/` structure have been removed. This repository now uses only the Version 2 microservices architecture.
