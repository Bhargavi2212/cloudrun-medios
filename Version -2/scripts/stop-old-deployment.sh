#!/bin/bash

# Stop and delete old Medi OS deployment from cloudrun-medios repository
# Usage: ./scripts/stop-old-deployment.sh [project-id] [region]

set -e

PROJECT_ID=${1:-${GCP_PROJECT_ID}}
REGION=${2:-us-central1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping old Medi OS deployment...${NC}"
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

# List of old services to delete
OLD_SERVICES=(
    "medios-backend"
    "medios-frontend"
)

echo -e "${YELLOW}Checking for existing services...${NC}"

# Delete each service if it exists
for SERVICE in "${OLD_SERVICES[@]}"; do
    echo -e "${YELLOW}Checking service: ${SERVICE}...${NC}"
    
    # Check if service exists
    if gcloud run services describe ${SERVICE} --region ${REGION} --format="value(metadata.name)" 2>/dev/null; then
        echo -e "${RED}Found service: ${SERVICE}${NC}"
        read -p "Delete ${SERVICE}? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Deleting ${SERVICE}...${NC}"
            gcloud run services delete ${SERVICE} --region ${REGION} --quiet || true
            echo -e "${GREEN}Deleted ${SERVICE}${NC}"
        else
            echo -e "${YELLOW}Skipping ${SERVICE}${NC}"
        fi
    else
        echo -e "${GREEN}Service ${SERVICE} not found (already deleted or never existed)${NC}"
    fi
done

echo -e "${GREEN}Old deployment cleanup complete!${NC}"
echo ""
echo -e "${YELLOW}Note: Cloud SQL instances and Cloud Storage buckets are NOT deleted.${NC}"
echo -e "${YELLOW}You may want to delete them manually if you no longer need them.${NC}"

