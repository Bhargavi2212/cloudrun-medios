# GCP Cloud Run Deployment Setup - Complete ✅

## What Was Created

### 1. Deployment Scripts

#### `scripts/deploy-cloud-run-backend.sh`
- Builds and deploys backend to Cloud Run
- Configures environment variables
- Connects to Cloud SQL
- Sets up secrets from Secret Manager
- Configures resource limits and scaling

#### `scripts/deploy-cloud-run-frontend.sh`
- Builds frontend with environment variables
- Deploys to Cloud Run
- Configures nginx for SPA routing
- Sets resource limits and scaling

#### `scripts/setup-cloud-sql.sh`
- Creates PostgreSQL 15 instance
- Creates database and user
- Stores password in Secret Manager
- Configures Cloud Run access
- Sets up connection name

#### `scripts/setup-cloud-storage.sh`
- Creates GCS bucket
- Configures lifecycle policies
- Sets up CORS for frontend access
- Creates folder structure
- Grants Cloud Run access

#### `scripts/setup-gcp-secrets.sh`
- Creates JWT secrets
- Stores Gemini API key
- Grants Cloud Run service account access
- Supports multiple environments

#### `scripts/deploy-all.sh`
- Orchestrates full deployment
- Runs all setup scripts in order
- Deploys backend and frontend
- Provides deployment summary

### 2. Documentation

#### `docs/deployment.md`
- Complete deployment guide
- Step-by-step instructions
- Troubleshooting guide
- Cost estimation
- Security best practices

## Quick Start

```bash
# Set environment variables
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1

# Make scripts executable (Linux/Mac)
chmod +x scripts/*.sh

# Deploy everything
./scripts/deploy-all.sh production
```

## Deployment Flow

1. **Setup Cloud SQL** → PostgreSQL database
2. **Setup Cloud Storage** → File storage bucket
3. **Setup Secrets** → JWT and API keys
4. **Deploy Backend** → FastAPI service
5. **Deploy Frontend** → React app

## Features

### ✅ Automated Setup
- All GCP resources created automatically
- APIs enabled automatically
- IAM permissions configured
- Secrets stored securely

### ✅ Environment Support
- Supports multiple environments (dev/staging/prod)
- Environment-specific resources
- Environment-specific secrets

### ✅ Security
- Secrets stored in Secret Manager
- Cloud SQL private IP (via Unix socket)
- IAM-based access control
- HTTPS enabled by default

### ✅ Scalability
- Auto-scaling Cloud Run services
- Configurable min/max instances
- Resource limits configured
- Connection pooling

## Configuration

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

Approximate monthly costs (us-central1, low-medium traffic):

- Cloud SQL (db-f1-micro): ~$7-10/month
- Cloud Run (backend): ~$20-50/month
- Cloud Run (frontend): ~$5-15/month
- Cloud Storage: ~$1-5/month
- Container Registry: ~$1-3/month

**Total**: ~$35-85/month

## Next Steps

1. **Run database migrations** after deployment
2. **Configure custom domain** for production
3. **Set up monitoring alerts** in Cloud Console
4. **Configure backup strategy** for Cloud SQL
5. **Set up CI/CD pipeline** (already created in `.github/workflows/`)

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check logs, verify environment variables
2. **Database connection issues**: Verify Cloud SQL instance and connection name
3. **Frontend not loading**: Check backend URL and CORS settings
4. **Secrets not accessible**: Verify IAM permissions on secrets

See `docs/deployment.md` for detailed troubleshooting guide.

