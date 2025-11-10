#!/bin/bash

# Deploy Medi OS Frontend to Google Cloud Run
# Usage: ./scripts/deploy-cloud-run-frontend.sh [environment] [project-id] [backend-url]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
BACKEND_URL=${3:-${BACKEND_URL}}
REGION=${GCP_REGION:-us-central1}
SERVICE_NAME="medios-frontend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying Medi OS Frontend to Cloud Run...${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Backend URL: ${BACKEND_URL}"
echo -e "Region: ${REGION}"
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set. Please provide project ID as argument or set GCP_PROJECT_ID environment variable.${NC}"
    exit 1
fi

# Check if backend URL is set
if [ -z "$BACKEND_URL" ]; then
    echo -e "${YELLOW}Warning: BACKEND_URL not set. Attempting to get from backend service...${NC}"
    BACKEND_URL=$(gcloud run services describe medios-backend --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo "")
    if [ -z "$BACKEND_URL" ]; then
        echo -e "${RED}Error: Could not determine backend URL. Please provide it as the third argument or set BACKEND_URL environment variable.${NC}"
        exit 1
    fi
fi

# Set the GCP project
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Determine WebSocket URL
WS_URL=$(echo ${BACKEND_URL} | sed 's|https://|wss://|' | sed 's|http://|ws://|')

# Build and push Docker image with build args
echo -e "${YELLOW}Building Docker image...${NC}"
cd frontend

# Create cloudbuild.yaml for build args
cat > cloudbuild.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--build-arg'
      - 'VITE_API_BASE_URL=${BACKEND_URL}/api/v1'
      - '--build-arg'
      - 'VITE_WS_URL=${WS_URL}/ws'
      - '--build-arg'
      - 'VITE_ENVIRONMENT=${ENVIRONMENT}'
      - '--tag'
      - '${IMAGE_NAME}:latest'
      - '--tag'
      - '${IMAGE_NAME}:${ENVIRONMENT}'
      - '.'
images:
  - '${IMAGE_NAME}:latest'
  - '${IMAGE_NAME}:${ENVIRONMENT}'
EOF

gcloud builds submit --config=cloudbuild.yaml

# Clean up
rm cloudbuild.yaml

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"

gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:${ENVIRONMENT} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 60 \
    --max-instances 5 \
    --min-instances 0 \
    --concurrency 1000 \
    --port 80

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "Frontend URL: ${SERVICE_URL}"
echo -e "Backend URL: ${BACKEND_URL}"
echo -e "Health check: ${SERVICE_URL}/health"

cd ..

