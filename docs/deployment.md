# Medi OS Deployment Guide

This guide covers deploying Medi OS to Google Cloud Run.

## Prerequisites

1. **Google Cloud Project**: Create a GCP project or use an existing one
2. **gcloud CLI**: Install and configure [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
3. **Authentication**: Authenticate with GCP:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
4. **Billing**: Enable billing on your GCP project
5. **APIs**: Required APIs will be enabled automatically by the scripts

## Quick Start

### Deploy Everything

```bash
# Set environment variables
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

# Make scripts executable
chmod +x scripts/*.sh

# Deploy everything
./scripts/deploy-all.sh production
```

This will:
1. Setup Cloud SQL database
2. Setup Cloud Storage bucket
3. Setup Secret Manager secrets
4. Deploy backend to Cloud Run
5. Deploy frontend to Cloud Run

## Step-by-Step Deployment

### 1. Setup Cloud SQL

```bash
./scripts/setup-cloud-sql.sh production
```

This creates:
- PostgreSQL 15 instance
- Database: `medios_db`
- User: `medios_user`
- Password stored in Secret Manager

### 2. Setup Cloud Storage

```bash
./scripts/setup-cloud-storage.sh production
```

This creates:
- Storage bucket: `{project-id}-medios-storage-production`
- Lifecycle policies for file retention
- CORS configuration for frontend access

### 3. Setup Secrets

```bash
# Set secrets (optional - will prompt if not set)
export JWT_ACCESS_SECRET=your-secret
export JWT_REFRESH_SECRET=your-secret
export GEMINI_API_KEY=your-key

./scripts/setup-gcp-secrets.sh production
```

This creates:
- JWT access secret
- JWT refresh secret
- Gemini API key (optional)

### 4. Deploy Backend

```bash
./scripts/deploy-cloud-run-backend.sh production
```

This:
- Builds Docker image
- Pushes to Container Registry
- Deploys to Cloud Run
- Configures environment variables
- Connects to Cloud SQL

### 5. Deploy Frontend

```bash
# Get backend URL first
BACKEND_URL=$(gcloud run services describe medios-backend --region us-central1 --format 'value(status.url)')

./scripts/deploy-cloud-run-frontend.sh production $GCP_PROJECT_ID $BACKEND_URL
```

This:
- Builds Docker image with environment variables
- Pushes to Container Registry
- Deploys to Cloud Run

## Database Migrations

After deploying, run database migrations:

```bash
# Option 1: Using Cloud Run Jobs
gcloud run jobs create medios-migrate \
  --image gcr.io/$GCP_PROJECT_ID/medios-backend:production \
  --region us-central1 \
  --set-env-vars DATABASE_URL="..." \
  --command "alembic" \
  --args "upgrade head"

# Option 2: Using Cloud SQL Proxy locally
cloud-sql-proxy $PROJECT_ID:us-central1:medios-db-production
alembic upgrade head
```

## Environment Variables

### Backend Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `APP_ENV` | Environment (production/staging/dev) | Yes |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `JWT_ACCESS_SECRET` | JWT access token secret | Yes |
| `JWT_REFRESH_SECRET` | JWT refresh token secret | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `STORAGE_BACKEND` | Storage backend (gcs) | Yes |
| `STORAGE_GCS_BUCKET` | GCS bucket name | Yes |
| `SECRET_MANAGER_ENABLED` | Enable Secret Manager | Yes |

### Frontend Build Arguments

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_API_BASE_URL` | Backend API URL | Yes |
| `VITE_WS_URL` | WebSocket URL | Yes |
| `VITE_ENVIRONMENT` | Environment | Yes |

## Cloud Run Configuration

### Backend Service

- **Memory**: 2Gi
- **CPU**: 2
- **Timeout**: 300s
- **Max Instances**: 10
- **Min Instances**: 1
- **Concurrency**: 80

### Frontend Service

- **Memory**: 512Mi
- **CPU**: 1
- **Timeout**: 60s
- **Max Instances**: 5
- **Min Instances**: 0
- **Concurrency**: 1000

## Cost Estimation

Approximate monthly costs (us-central1):

- **Cloud SQL (db-f1-micro)**: ~$7-10/month
- **Cloud Run (backend)**: ~$20-50/month (depends on usage)
- **Cloud Run (frontend)**: ~$5-15/month (depends on usage)
- **Cloud Storage**: ~$1-5/month (depends on storage)
- **Container Registry**: ~$1-3/month (depends on images)

**Total**: ~$35-85/month (low to medium traffic)

## Monitoring

### View Logs

```bash
# Backend logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=medios-backend" --limit 50

# Frontend logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=medios-frontend" --limit 50
```

### View Metrics

```bash
# Open Cloud Console
gcloud monitoring dashboards list
```

## Troubleshooting

### Backend Not Starting

1. Check logs: `gcloud run services logs read medios-backend`
2. Verify environment variables are set correctly
3. Check Cloud SQL connection
4. Verify secrets are accessible

### Frontend Not Loading

1. Check backend URL is correct
2. Verify CORS settings
3. Check nginx logs in Cloud Run
4. Verify build arguments were passed correctly

### Database Connection Issues

1. Verify Cloud SQL instance is running
2. Check Cloud SQL connection name is correct
3. Verify Cloud Run service account has Cloud SQL Client role
4. Check database credentials in Secret Manager

## Rolling Back

To roll back to a previous version:

```bash
# List revisions
gcloud run revisions list --service medios-backend

# Roll back to specific revision
gcloud run services update-traffic medios-backend \
  --to-revisions REVISION_NAME=100
```

## Cleanup

To remove all resources:

```bash
# Delete Cloud Run services
gcloud run services delete medios-backend --region us-central1
gcloud run services delete medios-frontend --region us-central1

# Delete Cloud SQL instance
gcloud sql instances delete medios-db-production

# Delete Cloud Storage bucket
gsutil rm -r gs://$PROJECT_ID-medios-storage-production

# Delete secrets
gcloud secrets delete jwt-access-secret-production
gcloud secrets delete jwt-refresh-secret-production
gcloud secrets delete gemini-api-key
```

## Security Best Practices

1. **Use Secret Manager** for all secrets
2. **Enable VPC** for Cloud SQL (optional, for production)
3. **Use IAM** to restrict access
4. **Enable audit logs** for compliance
5. **Use HTTPS** only (enabled by default on Cloud Run)
6. **Rotate secrets** regularly
7. **Use least privilege** IAM roles

## Next Steps

1. Set up custom domain
2. Configure CDN for frontend
3. Set up monitoring alerts
4. Configure backup strategy for Cloud SQL
5. Set up CI/CD pipeline (see `.github/workflows/ci-cd.yml`)

