# Deploy Medi OS v2 Now - Step by Step

**Your Project ID:** `plasma-datum-476821-s8`

## Prerequisites Check

```bash
# 1. Verify you're authenticated
gcloud auth list

# 2. Set your project
gcloud config set project plasma-datum-476821-s8

# 3. Verify billing is enabled
gcloud billing projects describe plasma-datum-476821-s8
```

## Step 1: Stop Old Services

```bash
export GCP_PROJECT_ID=plasma-datum-476821-s8
export GCP_REGION=us-central1

# Run the stop script
bash scripts/stop-old-deployment.sh plasma-datum-476821-s8 us-central1
```

**What this does:**
- Finds old `medios-backend` and `medios-frontend` services
- Asks you to confirm deletion for each
- **Does NOT delete:** Cloud SQL, Cloud Storage, or Secrets

## Step 2: Setup Secrets (One-time)

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe plasma-datum-476821-s8 --format="value(projectNumber)")
echo "Project Number: ${PROJECT_NUMBER}"

# Create secrets (replace with your actual values)
echo -n "your-dol-secret-here" | gcloud secrets create DOL_SHARED_SECRET --data-file=-
echo -n "your-federation-secret-here" | gcloud secrets create FEDERATION_SHARED_SECRET --data-file=-
echo -n "AIzaSyD3R7WCLviEOxz8oFkr1uFbZK7Nibe4Xuo" | gcloud secrets create GEMINI_API_KEY --data-file=-

# Grant Cloud Run service account access
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

**Note:** If secrets already exist, you'll get an error. That's okay - skip to the next step.

## Step 3: Deploy All Services

```bash
# Make sure you're in the Version -2 directory
cd "Version -2"

# Deploy everything
bash scripts/deploy-all-services.sh production plasma-datum-476821-s8 us-central1
```

**This will take 10-15 minutes** as it:
1. Builds Docker images for each service
2. Pushes to Container Registry
3. Deploys to Cloud Run
4. Configures environment variables
5. Sets up service URLs

## Step 4: Get Your URLs

After deployment completes, get your service URLs:

```bash
# Frontend URL (main entry point)
gcloud run services describe medios-frontend-production \
    --region=us-central1 \
    --project=plasma-datum-476821-s8 \
    --format='value(status.url)'

# List all services
gcloud run services list --region=us-central1 --project=plasma-datum-476821-s8
```

## Step 5: Test Your Deployment

```bash
# Test health endpoints
curl https://manage-agent-production-us-central1-plasma-datum-476821-s8.a.run.app/health
curl https://dol-service-production-us-central1-plasma-datum-476821-s8.a.run.app/health
```

## Troubleshooting

### If deployment fails:

1. **Check logs:**
   ```bash
   gcloud builds list --limit=5 --project=plasma-datum-476821-s8
   ```

2. **Check service status:**
   ```bash
   gcloud run services list --region=us-central1 --project=plasma-datum-476821-s8
   ```

3. **View service logs:**
   ```bash
   gcloud run services logs read manage-agent-production \
       --region=us-central1 \
       --project=plasma-datum-476821-s8
   ```

### Common Issues:

- **"Permission denied"**: Make sure you're authenticated and have proper IAM roles
- **"Secret not found"**: Run Step 2 to create secrets
- **"Cloud SQL connection failed"**: Check if Cloud SQL instance exists and is accessible

## Next Steps

1. ✅ Update frontend configuration with production URLs
2. ✅ Run database migrations
3. ✅ Test the full system
4. ✅ Set up monitoring

## Quick Commands Reference

```bash
# Set project
gcloud config set project plasma-datum-476821-s8

# List services
gcloud run services list --region=us-central1

# View logs
gcloud run services logs read SERVICE_NAME --region=us-central1

# Delete a service
gcloud run services delete SERVICE_NAME --region=us-central1
```

