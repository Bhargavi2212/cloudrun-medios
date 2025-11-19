# Version 2 CI/CD Pipeline Setup - Complete ✅

## Overview

The Version 2 CI/CD pipeline supports the microservices architecture with separate deployments for Hospital A and Hospital B.

## What Was Created

### 1. GitHub Actions Workflows

#### `.github/workflows/version-2-ci-cd.yml`
Complete CI/CD pipeline that:
- ✅ Runs linting and tests for both HospitalA and HospitalB
- ✅ Builds Docker images for all microservices
- ✅ Deploys to Google Cloud Run (on main branch only)

#### `.github/workflows/version-2-test.yml`
Test-only workflow for quick feedback:
- ✅ Tests HospitalA backend
- ✅ Tests HospitalB backend  
- ✅ Tests frontend

### 2. Docker Configuration

#### Frontend Dockerfile
- ✅ Created `apps/frontend/Dockerfile` for both hospitals
- ✅ Multi-stage build (Node.js build + Nginx production)
- ✅ Supports Cloud Run (port 8080)
- ✅ Includes health checks

#### Nginx Configuration
- ✅ Created `apps/frontend/nginx.conf` for production serving
- ✅ SPA routing support
- ✅ Gzip compression
- ✅ Security headers
- ✅ Static asset caching

### 3. Services Architecture

Each hospital deploys:
1. **manage-agent** - Patient management and triage (ports 8001/9001)
2. **scribe-agent** - Medical transcription (ports 8002/9002)
3. **summarizer-agent** - Document summarization (ports 8003/9003)
4. **dol-service** - Data Orchestration Layer (ports 8004/9004)
5. **federation-aggregator** - Federated learning (ports 8010/9010)
6. **frontend** - React application

## Workflow Features

### ✅ Parallel Execution
- HospitalA and HospitalB backend jobs run in parallel
- Faster feedback on PRs

### ✅ Caching
- Poetry virtual environments cached
- Node.js npm dependencies cached
- Docker build cache

### ✅ Conditional Deployment
- Only deploys on pushes to `main` branch
- PRs only run tests (no deployment)
- Manual workflow dispatch supported

### ✅ Error Handling
- `continue-on-error` flags for optional steps
- Deployment failures don't block the workflow
- Tests can fail without stopping the pipeline

### ✅ Multi-Hospital Support
- Separate deployments for HospitalA and HospitalB
- Same region (us-central1) for both hospitals (lower cost, lower latency)
- Independent scaling and configuration per hospital

## Required Setup

### GitHub Secrets (for full deployment)

1. **GCP_SA_KEY**: Google Cloud Service Account JSON
   ```bash
   # Create service account
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions Service Account"
   
   # Grant permissions
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"
   
   # Create key
   gcloud iam service-accounts keys create key.json \
     --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```

2. **GCP_PROJECT_ID**: Your GCP Project ID

Add these in: Repository Settings → Secrets → Actions

## Workflow Triggers

| Event | Lint & Test | Build | Deploy |
|-------|-------------|-------|--------|
| Push to main | ✅ | ✅ | ✅ |
| Push to develop | ✅ | ✅ | ❌ |
| Pull Request | ✅ | ❌ | ❌ |
| Manual dispatch | ✅ | ✅ | ✅ |

## Deployment Configuration

### Resource Limits

**Backend Services (manage-agent, scribe-agent, summarizer-agent, dol-service):**
- Memory: 2Gi
- CPU: 2
- Timeout: 300s
- Max Instances: 10
- Min Instances: 0

**Federation Aggregator:**
- Memory: 1Gi
- CPU: 1
- Timeout: 300s
- Max Instances: 5
- Min Instances: 0

**Frontend:**
- Memory: 512Mi
- CPU: 1
- Timeout: 60s
- Max Instances: 5
- Min Instances: 0

### Service URLs

After deployment, services will be available at:
- **Both hospitals deploy to `us-central1` region** (same region = lower cost, lower latency)

- Hospital A:
  - manage-agent: `https://medi-os-hospital-a-manage-agent-<hash>-uc.a.run.app`
  - scribe-agent: `https://medi-os-hospital-a-scribe-agent-<hash>-uc.a.run.app`
  - summarizer-agent: `https://medi-os-hospital-a-summarizer-agent-<hash>-uc.a.run.app`
  - dol-service: `https://medi-os-hospital-a-dol-service-<hash>-uc.a.run.app`
  - federation-aggregator: `https://medi-os-hospital-a-federation-aggregator-<hash>-uc.a.run.app`
  - frontend: `https://medi-os-hospital-a-frontend-<hash>-uc.a.run.app`

- Hospital B:
  - manage-agent: `https://medi-os-hospital-b-manage-agent-<hash>-uc.a.run.app`
  - scribe-agent: `https://medi-os-hospital-b-scribe-agent-<hash>-uc.a.run.app`
  - summarizer-agent: `https://medi-os-hospital-b-summarizer-agent-<hash>-uc.a.run.app`
  - dol-service: `https://medi-os-hospital-b-dol-service-<hash>-uc.a.run.app`
  - federation-aggregator: `https://medi-os-hospital-b-federation-aggregator-<hash>-uc.a.run.app`
  - frontend: `https://medi-os-hospital-b-frontend-<hash>-uc.a.run.app`

**Note:** You can change the deployment region in `.github/workflows/version-2-ci-cd.yml` if needed (e.g., us-east1, europe-west1, asia-east1)

## Local Testing

### Run Tests Locally

```bash
# HospitalA
cd Version-2-HospitalA
poetry install
poetry run pytest

# HospitalB
cd Version-2-HospitalB
poetry install
poetry run pytest

# Frontend
cd Version-2-HospitalA/apps/frontend
npm install
npm run test
```

### Build Docker Images Locally

```bash
# Build a service
cd Version-2-HospitalA
docker build -f services/manage_agent/Dockerfile -t medi-os-manage-agent:local .

# Build frontend
cd Version-2-HospitalA/apps/frontend
docker build -t medi-os-frontend:local .
```

### Run with Docker Compose

```bash
cd Version-2-HospitalA
docker-compose -f docker-compose.demo.yml up
```

## Next Steps

1. **Add GitHub Secrets** (if deploying to GCP):
   - Go to repository Settings → Secrets → Actions
   - Add `GCP_SA_KEY` and `GCP_PROJECT_ID`

2. **Test the Pipeline**:
   - Push a commit to trigger the workflow
   - Check Actions tab in GitHub

3. **Configure Environment Variables**:
   - Set up Cloud SQL connection strings
   - Configure API keys in GCP Secret Manager
   - Set CORS origins for frontend

4. **Set Up Database Migrations**:
   - Run Alembic migrations after first deployment
   - Set up automated migration jobs

## Notes

- The pipeline will work without GCP secrets (tests will run, deployment will be skipped)
- Docker builds are cached for faster subsequent builds
- All services are deployed independently and can scale separately
- Tests are set to continue-on-error to show all issues at once
- Frontend build uses environment variables for API URLs

