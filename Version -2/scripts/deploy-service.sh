#!/bin/bash

# Deploy a single service to Google Cloud Run
# Usage: ./scripts/deploy-service.sh [service-name] [environment] [project-id] [region]

set -e

SERVICE_NAME=${1}
ENVIRONMENT=${2:-production}
PROJECT_ID=${3:-${GCP_PROJECT_ID}}
REGION=${4:-us-central1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ -z "$SERVICE_NAME" ]; then
    echo -e "${RED}Error: Service name is required${NC}"
    echo "Usage: ./scripts/deploy-service.sh [service-name] [environment] [project-id] [region]"
    echo "Services: manage-agent, scribe-agent, summarizer-agent, dol-service, federation"
    exit 1
fi

# Service configuration mapping
declare -A SERVICE_PORTS
SERVICE_PORTS[manage-agent]=8001
SERVICE_PORTS[scribe-agent]=8002
SERVICE_PORTS[summarizer-agent]=8003
SERVICE_PORTS[dol-service]=8004
SERVICE_PORTS[federation]=8010

declare -A SERVICE_PATHS
SERVICE_PATHS[manage-agent]="services/manage_agent"
SERVICE_PATHS[scribe-agent]="services/scribe_agent"
SERVICE_PATHS[summarizer-agent]="services/summarizer_agent"
SERVICE_PATHS[dol-service]="dol_service"
SERVICE_PATHS[federation]="federation"

# Validate service name
if [ -z "${SERVICE_PATHS[$SERVICE_NAME]}" ]; then
    echo -e "${RED}Error: Invalid service name: ${SERVICE_NAME}${NC}"
    echo "Valid services: manage-agent, scribe-agent, summarizer-agent, dol-service, federation"
    exit 1
fi

SERVICE_PATH=${SERVICE_PATHS[$SERVICE_NAME]}
SERVICE_PORT=${SERVICE_PORTS[$SERVICE_NAME]}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
CLOUD_RUN_SERVICE="${SERVICE_NAME}-${ENVIRONMENT}"

echo -e "${GREEN}Deploying ${SERVICE_NAME} to Cloud Run...${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo -e "Port: ${SERVICE_PORT}"
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
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable sqladmin.googleapis.com --quiet
gcloud services enable secretmanager.googleapis.com --quiet

# Build Docker image
echo -e "${YELLOW}Building Docker image for ${SERVICE_NAME}...${NC}"

# Get the root directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Build from root directory with proper context
cd "${ROOT_DIR}"

# Use Cloud Build to build the Docker image
# The Dockerfile is in the service directory, but we need root context for shared files
gcloud builds submit \
    --tag ${IMAGE_NAME}:${ENVIRONMENT} \
    --tag ${IMAGE_NAME}:latest \
    --config <(cat <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: 
      - 'build'
      - '-f'
      - '${SERVICE_PATH}/Dockerfile'
      - '-t'
      - '${IMAGE_NAME}:${ENVIRONMENT}'
      - '-t'
      - '${IMAGE_NAME}:latest'
      - '.'
images:
  - '${IMAGE_NAME}:${ENVIRONMENT}'
  - '${IMAGE_NAME}:latest'
EOF
) \
    . || {
    echo -e "${RED}Error: Failed to build Docker image${NC}"
    exit 1
}

cd - > /dev/null

# Get Cloud SQL connection name (if exists)
CLOUD_SQL_INSTANCE=$(gcloud sql instances list --filter="name:medios-db-${ENVIRONMENT}" --format="value(connectionName)" --limit=1 2>/dev/null || echo "")

# Prepare environment variables
ENV_VARS=(
    "APP_ENV=${ENVIRONMENT}"
    "DATABASE_URL=postgresql+asyncpg://postgres:password@/medios_db?host=/cloudsql/${CLOUD_SQL_INSTANCE}"
)

# Add service-specific environment variables
case $SERVICE_NAME in
    manage-agent)
        ENV_VARS+=(
            "DOL_BASE_URL=https://dol-service-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
            "DOL_SHARED_SECRET=\$(DOL_SHARED_SECRET:latest)"
        )
        ;;
    scribe-agent)
        ENV_VARS+=(
            "GEMINI_API_KEY=\$(GEMINI_API_KEY:latest)"
        )
        ;;
    summarizer-agent)
        ENV_VARS+=(
            "GEMINI_API_KEY=\$(GEMINI_API_KEY:latest)"
            "SUMMARIZER_AGENT_CORS_ORIGINS=https://medios-frontend-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
        )
        ;;
    dol-service)
        ENV_VARS+=(
            "DOL_SHARED_SECRET=\$(DOL_SHARED_SECRET:latest)"
        )
        ;;
    federation)
        ENV_VARS+=(
            "FEDERATION_SHARED_SECRET=\$(FEDERATION_SHARED_SECRET:latest)"
        )
        ;;
esac

# Build environment variables string
ENV_VARS_STRING=$(IFS=,; echo "${ENV_VARS[*]}")

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying ${SERVICE_NAME} to Cloud Run...${NC}"

DEPLOY_CMD="gcloud run deploy ${CLOUD_RUN_SERVICE} \
    --image ${IMAGE_NAME}:${ENVIRONMENT} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --set-env-vars ${ENV_VARS_STRING} \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --min-instances 0 \
    --concurrency 80 \
    --port ${SERVICE_PORT}"

# Add Cloud SQL connection if available
if [ -n "$CLOUD_SQL_INSTANCE" ]; then
    DEPLOY_CMD="${DEPLOY_CMD} --add-cloudsql-instances ${CLOUD_SQL_INSTANCE}"
fi

eval $DEPLOY_CMD

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${CLOUD_RUN_SERVICE} --region ${REGION} --format 'value(status.url)')

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "Service: ${SERVICE_NAME}"
echo -e "Service URL: ${SERVICE_URL}"
echo -e "Health check: ${SERVICE_URL}/health"

