#!/bin/bash

# Deploy Medi OS Backend to Google Cloud Run
# Usage: ./scripts/deploy-cloud-run-backend.sh [environment] [project-id]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${GCP_REGION:-us-central1}
SERVICE_NAME="medios-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying Medi OS Backend to Cloud Run...${NC}"
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

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Build and push Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd backend
gcloud builds submit --tag ${IMAGE_NAME}:latest --tag ${IMAGE_NAME}:${ENVIRONMENT}

# Get Cloud SQL connection name
CLOUD_SQL_INSTANCE=$(gcloud sql instances describe medios-db-${ENVIRONMENT} --format="value(connectionName)" 2>/dev/null || echo "")
if [ -z "$CLOUD_SQL_INSTANCE" ]; then
    echo -e "${YELLOW}Warning: Cloud SQL instance not found. Please create it using scripts/setup-cloud-sql.sh${NC}"
    echo -e "${RED}Error: Cannot deploy without Cloud SQL instance.${NC}"
    exit 1
fi

# Get database password from Secret Manager
DB_PASSWORD=$(gcloud secrets versions access latest --secret=db-password-${ENVIRONMENT} 2>/dev/null || echo "")
if [ -z "$DB_PASSWORD" ]; then
    echo -e "${YELLOW}Warning: Database password not found in Secret Manager. Using Cloud SQL Unix socket connection.${NC}"
    DATABASE_URL="postgresql+psycopg2://medios_user@/medios_db?host=/cloudsql/${CLOUD_SQL_INSTANCE}"
else
    DATABASE_URL="postgresql+psycopg2://medios_user:${DB_PASSWORD}@/medios_db?host=/cloudsql/${CLOUD_SQL_INSTANCE}"
fi

# Prepare environment variables
ENV_VARS=(
    "APP_ENV=${ENVIRONMENT}"
    "APP_DEBUG=false"
    "DATABASE_URL=${DATABASE_URL}"
    "STORAGE_BACKEND=gcs"
    "STORAGE_GCS_BUCKET=${PROJECT_ID}-medios-storage-${ENVIRONMENT}"
    "SECRET_MANAGER_ENABLED=true"
    "SECRET_MANAGER_PROJECT_ID=${PROJECT_ID}"
    "SECRET_MANAGER_ENVIRONMENT=${ENVIRONMENT}"
)

# Add secrets from Secret Manager
SECRETS=(
    "JWT_ACCESS_SECRET:jwt-access-secret-${ENVIRONMENT}"
    "JWT_REFRESH_SECRET:jwt-refresh-secret-${ENVIRONMENT}"
    "GEMINI_API_KEY:gemini-api-key"
)

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"

# Build environment variables string
ENV_VARS_STRING=$(IFS=,; echo "${ENV_VARS[*]}")

# Build secrets string
SECRETS_STRING=$(IFS=,; printf '%s\n' "${SECRETS[@]}" | sed 's/:/:projects\/'"${PROJECT_ID}"'\/secrets\//' | tr '\n' ',' | sed 's/,$//')

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:${ENVIRONMENT} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --set-env-vars ${ENV_VARS_STRING} \
    --set-secrets ${SECRETS_STRING} \
    --add-cloudsql-instances ${CLOUD_SQL_INSTANCE} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 1 \
    --concurrency 80 \
    --port 8080

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "Service URL: ${SERVICE_URL}"
echo -e "Health check: ${SERVICE_URL}/health"

cd ..

