#!/bin/bash

# Deploy all Medi OS v2 services to Google Cloud Run
# Usage: ./scripts/deploy-all-services.sh [environment] [project-id] [region]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${3:-us-central1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Medi OS v2 Deployment to Cloud Run${NC}"
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

# Service deployment order (dependencies first)
SERVICES=(
    "dol-service"          # DOL orchestrator (no dependencies)
    "federation"           # Federation aggregator (no dependencies)
    "manage-agent"         # Manage agent (depends on DOL)
    "scribe-agent"         # Scribe agent (independent)
    "summarizer-agent"     # Summarizer agent (independent)
)

# Deploy each service
for SERVICE in "${SERVICES[@]}"; do
    echo -e "${GREEN}Deploying ${SERVICE}...${NC}"
    ./scripts/deploy-service.sh ${SERVICE} ${ENVIRONMENT} ${PROJECT_ID} ${REGION}
    echo ""
done

# Deploy frontend (after all backends are up)
echo -e "${GREEN}Deploying frontend...${NC}"
./scripts/deploy-frontend.sh ${ENVIRONMENT} ${PROJECT_ID} ${REGION}

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All services deployed!${NC}"
echo -e "${BLUE}========================================${NC}"

# Get service URLs
echo -e "${YELLOW}Service URLs:${NC}"
for SERVICE in "${SERVICES[@]}"; do
    SERVICE_NAME="${SERVICE}-${ENVIRONMENT}"
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo "Not deployed")
    echo -e "  ${SERVICE}: ${SERVICE_URL}"
done

FRONTEND_URL=$(gcloud run services describe medios-frontend-${ENVIRONMENT} --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo "Not deployed")
echo -e "  Frontend: ${FRONTEND_URL}"

