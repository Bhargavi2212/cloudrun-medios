# Medi OS v2 - Google Cloud Run Deployment Guide

This guide covers deploying the new multi-service Medi OS v2 architecture to Google Cloud Run.

## Architecture Overview

Medi OS v2 consists of:
- **5 Backend Services:**
  - `manage-agent` (Port 8001) - Patient management and triage
  - `scribe-agent` (Port 8002) - SOAP note generation
  - `summarizer-agent` (Port 8003) - Patient summary generation
  - `dol-service` (Port 8004) - Data Orchestration Layer (DOL)
  - `federation` (Port 8010) - Federated learning aggregator
- **1 Frontend Service:**
  - React SPA with multi-hospital support

## Prerequisites

1. **Google Cloud Project**: Create or use an existing GCP project
2. **gcloud CLI**: Install and configure [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. **Authentication**:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
4. **Billing**: Enable billing on your GCP project
5. **APIs**: Required APIs will be enabled automatically by the scripts

## Step 1: Stop Old Deployment

Before deploying the new version, stop the old deployment:

```bash
# Set your project ID
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

# Make scripts executable
chmod +x scripts/*.sh

# Stop old services
./scripts/stop-old-deployment.sh
```

This will:
- List all old Cloud Run services
- Prompt you to delete each one
- **Note**: Cloud SQL and Cloud Storage are NOT deleted (you may want to keep them)

## Step 2: Setup Infrastructure

### 2.1 Setup Cloud SQL Database

```bash
# Create PostgreSQL instance (if not exists)
gcloud sql instances create medios-db-production \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --root-password=your-secure-password

# Create database
gcloud sql databases create medios_db --instance=medios-db-production

# Create user
gcloud sql users create medios_user \
    --instance=medios-db-production \
    --password=your-secure-password
```

### 2.2 Setup Secret Manager

```bash
# Create secrets
echo -n "your-dol-secret" | gcloud secrets create DOL_SHARED_SECRET --data-file=-
echo -n "your-federation-secret" | gcloud secrets create FEDERATION_SHARED_SECRET --data-file=-
echo -n "your-gemini-api-key" | gcloud secrets create GEMINI_API_KEY --data-file=-

# Grant Cloud Run access
gcloud secrets add-iam-policy-binding DOL_SHARED_SECRET \
    --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Step 3: Deploy Services

### Option A: Deploy All Services at Once

```bash
# Deploy everything
./scripts/deploy-all-services.sh production
```

This will deploy services in the correct order:
1. DOL service (no dependencies)
2. Federation aggregator (no dependencies)
3. Manage agent (depends on DOL)
4. Scribe agent (independent)
5. Summarizer agent (independent)
6. Frontend (depends on all backends)

### Option B: Deploy Services Individually

```bash
# Deploy each service
./scripts/deploy-service.sh dol-service production
./scripts/deploy-service.sh federation production
./scripts/deploy-service.sh manage-agent production
./scripts/deploy-service.sh scribe-agent production
./scripts/deploy-service.sh summarizer-agent production

# Deploy frontend
./scripts/deploy-frontend.sh production
```

## Step 4: Verify Deployment

### Check Service URLs

```bash
# List all services
gcloud run services list --region=us-central1

# Get service URLs
gcloud run services describe manage-agent-production --region=us-central1 --format='value(status.url)'
```

### Test Health Endpoints

```bash
# Test each service
curl https://manage-agent-production-REGION-PROJECT_ID.a.run.app/health
curl https://scribe-agent-production-REGION-PROJECT_ID.a.run.app/health
curl https://summarizer-agent-production-REGION-PROJECT_ID.a.run.app/health
curl https://dol-service-production-REGION-PROJECT_ID.a.run.app/health
curl https://federation-production-REGION-PROJECT_ID.a.run.app/health
```

## Step 5: Update Frontend Configuration

After deployment, update the frontend to use the new service URLs:

1. Update `apps/frontend/src/shared/config/hospitalConfig.ts` with production URLs
2. Redeploy frontend:
   ```bash
   ./scripts/deploy-frontend.sh production
   ```

## Environment Variables

Each service requires specific environment variables:

### Manage Agent
- `DATABASE_URL` - PostgreSQL connection string
- `DOL_BASE_URL` - DOL service URL
- `DOL_SHARED_SECRET` - Secret from Secret Manager

### Scribe Agent
- `DATABASE_URL` - PostgreSQL connection string
- `GEMINI_API_KEY` - Secret from Secret Manager

### Summarizer Agent
- `DATABASE_URL` - PostgreSQL connection string
- `GEMINI_API_KEY` - Secret from Secret Manager
- `SUMMARIZER_AGENT_CORS_ORIGINS` - Frontend URL

### DOL Service
- `DATABASE_URL` - PostgreSQL connection string
- `DOL_SHARED_SECRET` - Secret from Secret Manager

### Federation
- `DATABASE_URL` - PostgreSQL connection string
- `FEDERATION_SHARED_SECRET` - Secret from Secret Manager

## Database Migrations

After deploying services, run database migrations:

```bash
# Connect to Cloud SQL and run migrations
gcloud sql connect medios-db-production --user=medios_user

# Or use Cloud Build to run migrations
# (Create a migration script that connects via Cloud SQL proxy)
```

## Troubleshooting

### Service Not Starting

1. Check logs:
   ```bash
   gcloud run services logs read manage-agent-production --region=us-central1
   ```

2. Check service status:
   ```bash
   gcloud run services describe manage-agent-production --region=us-central1
   ```

### Database Connection Issues

1. Verify Cloud SQL instance exists
2. Check Cloud SQL connection name format
3. Verify service account has Cloud SQL Client role

### Secret Access Issues

1. Verify secrets exist:
   ```bash
   gcloud secrets list
   ```

2. Check IAM permissions:
   ```bash
   gcloud secrets get-iam-policy DOL_SHARED_SECRET
   ```

## Cost Estimation

Approximate monthly costs (varies by usage):
- Cloud Run: ~$20-50/month (depending on traffic)
- Cloud SQL: ~$10-30/month (db-f1-micro)
- Cloud Storage: ~$5-10/month
- Secret Manager: ~$1/month

**Total: ~$36-91/month** (for low-medium traffic)

## Cleanup

To delete all services:

```bash
# Delete all Cloud Run services
gcloud run services delete manage-agent-production --region=us-central1
gcloud run services delete scribe-agent-production --region=us-central1
gcloud run services delete summarizer-agent-production --region=us-central1
gcloud run services delete dol-service-production --region=us-central1
gcloud run services delete federation-production --region=us-central1
gcloud run services delete medios-frontend-production --region=us-central1

# Delete Cloud SQL (optional - be careful!)
gcloud sql instances delete medios-db-production
```

## Next Steps

1. Set up CI/CD pipeline (GitHub Actions)
2. Configure custom domains
3. Set up monitoring and alerting
4. Configure backup strategies
5. Set up staging environment

## Support

For issues or questions:
- Check service logs in Cloud Console
- Review deployment scripts for configuration
- Verify all environment variables are set correctly

