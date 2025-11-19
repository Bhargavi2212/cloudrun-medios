# Quick Deployment Guide - Medi OS v2

**Project ID:** `plasma-datum-476821-s8`  
**Region:** `us-central1` (default)

## Step 1: Stop Old Deployment

```bash
# Set environment variables
export GCP_PROJECT_ID=plasma-datum-476821-s8
export GCP_REGION=us-central1

# Stop old services
bash scripts/stop-old-deployment.sh
```

This will:
- Find old `medios-backend` and `medios-frontend` services
- Prompt you to delete each one
- **Note:** Cloud SQL and Storage are NOT deleted (you may want to keep them)

## Step 2: Setup Secrets (if not already done)

```bash
# Set your project
gcloud config set project plasma-datum-476821-s8

# Create secrets (replace with your actual values)
echo -n "your-dol-secret-key" | gcloud secrets create DOL_SHARED_SECRET --data-file=-
echo -n "your-federation-secret-key" | gcloud secrets create FEDERATION_SHARED_SECRET --data-file=-
echo -n "your-gemini-api-key" | gcloud secrets create GEMINI_API_KEY --data-file=-

# Grant Cloud Run access (get PROJECT_NUMBER first)
PROJECT_NUMBER=$(gcloud projects describe plasma-datum-476821-s8 --format="value(projectNumber)")

gcloud secrets add-iam-policy-binding DOL_SHARED_SECRET \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding FEDERATION_SHARED_SECRET \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Step 3: Deploy All Services

```bash
# Deploy everything
bash scripts/deploy-all-services.sh production plasma-datum-476821-s8 us-central1
```

This will deploy:
1. ✅ DOL service
2. ✅ Federation aggregator
3. ✅ Manage agent
4. ✅ Scribe agent
5. ✅ Summarizer agent
6. ✅ Frontend

## Step 4: Verify Deployment

```bash
# List all services
gcloud run services list --region=us-central1 --project=plasma-datum-476821-s8

# Get frontend URL
gcloud run services describe medios-frontend-production --region=us-central1 --project=plasma-datum-476821-s8 --format='value(status.url)'
```

## Troubleshooting

### Check Service Logs
```bash
gcloud run services logs read manage-agent-production --region=us-central1 --project=plasma-datum-476821-s8
```

### Check Service Status
```bash
gcloud run services describe manage-agent-production --region=us-central1 --project=plasma-datum-476821-s8
```

## Service URLs (after deployment)

Your services will be available at:
- Manage Agent: `https://manage-agent-production-us-central1-plasma-datum-476821-s8.a.run.app`
- Scribe Agent: `https://scribe-agent-production-us-central1-plasma-datum-476821-s8.a.run.app`
- Summarizer Agent: `https://summarizer-agent-production-us-central1-plasma-datum-476821-s8.a.run.app`
- DOL Service: `https://dol-service-production-us-central1-plasma-datum-476821-s8.a.run.app`
- Federation: `https://federation-production-us-central1-plasma-datum-476821-s8.a.run.app`
- Frontend: `https://medios-frontend-production-us-central1-plasma-datum-476821-s8.a.run.app`

