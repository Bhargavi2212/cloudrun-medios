#!/bin/bash

# Deploy entire Medi OS stack to Google Cloud Run
# Usage: ./scripts/deploy-all.sh [environment] [project-id]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${GCP_REGION:-us-central1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Medi OS Deployment to Cloud Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set. Please provide project ID as argument or set GCP_PROJECT_ID environment variable.${NC}"
    exit 1
fi

# Set the GCP project
gcloud config set project ${PROJECT_ID}

# Step 1: Setup Cloud SQL
echo -e "${GREEN}Step 1: Setting up Cloud SQL...${NC}"
./scripts/setup-cloud-sql.sh ${ENVIRONMENT} ${PROJECT_ID}

# Step 2: Setup Cloud Storage
echo -e "${GREEN}Step 2: Setting up Cloud Storage...${NC}"
./scripts/setup-cloud-storage.sh ${ENVIRONMENT} ${PROJECT_ID}

# Step 3: Setup Secrets
echo -e "${GREEN}Step 3: Setting up Secret Manager...${NC}"
./scripts/setup-gcp-secrets.sh ${ENVIRONMENT} ${PROJECT_ID}

# Step 4: Deploy Backend
echo -e "${GREEN}Step 4: Deploying Backend...${NC}"
./scripts/deploy-cloud-run-backend.sh ${ENVIRONMENT} ${PROJECT_ID}

# Get backend URL
BACKEND_URL=$(gcloud run services describe medios-backend --region ${REGION} --format 'value(status.url)')

# Step 5: Deploy Frontend
echo -e "${GREEN}Step 5: Deploying Frontend...${NC}"
./scripts/deploy-cloud-run-frontend.sh ${ENVIRONMENT} ${PROJECT_ID} ${BACKEND_URL}

# Get frontend URL
FRONTEND_URL=$(gcloud run services describe medios-frontend --region ${REGION} --format 'value(status.url)')

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Frontend URL: ${FRONTEND_URL}"
echo -e "Backend URL: ${BACKEND_URL}"
echo -e "API Health: ${BACKEND_URL}/health"
echo -e "Frontend Health: ${FRONTEND_URL}/health"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Run database migrations:"
echo -e "   gcloud run jobs create medios-migrate --image gcr.io/${PROJECT_ID}/medios-backend:${ENVIRONMENT}"
echo -e "2. Update DNS to point to frontend URL"
echo -e "3. Configure CORS in backend if needed"

