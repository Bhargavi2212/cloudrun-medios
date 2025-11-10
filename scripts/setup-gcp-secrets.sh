#!/bin/bash

# Setup Google Cloud Secret Manager secrets for Medi OS
# Usage: ./scripts/setup-gcp-secrets.sh [environment] [project-id]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Secret Manager secrets...${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set. Please provide project ID as argument or set GCP_PROJECT_ID environment variable.${NC}"
    exit 1
fi

# Set the GCP project
gcloud config set project ${PROJECT_ID}

# Enable Secret Manager API
echo -e "${YELLOW}Enabling Secret Manager API...${NC}"
gcloud services enable secretmanager.googleapis.com

# Function to create or update secret
create_secret() {
    local secret_name=$1
    local secret_value=$2
    local description=$3
    
    if gcloud secrets describe ${secret_name} &>/dev/null; then
        echo -e "${YELLOW}Secret ${secret_name} already exists. Adding new version...${NC}"
        echo -n "${secret_value}" | gcloud secrets versions add ${secret_name} --data-file=-
    else
        echo -e "${YELLOW}Creating secret ${secret_name}...${NC}"
        echo -n "${secret_value}" | gcloud secrets create ${secret_name} \
            --data-file=- \
            --replication-policy="automatic" \
            --labels=environment=${ENVIRONMENT}
    fi
}

# JWT Secrets
if [ -z "$JWT_ACCESS_SECRET" ]; then
    JWT_ACCESS_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
    echo -e "${YELLOW}Generated JWT access secret${NC}"
fi

if [ -z "$JWT_REFRESH_SECRET" ]; then
    JWT_REFRESH_SECRET=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
    echo -e "${YELLOW}Generated JWT refresh secret${NC}"
fi

create_secret "jwt-access-secret-${ENVIRONMENT}" "${JWT_ACCESS_SECRET}" "JWT access token secret"
create_secret "jwt-refresh-secret-${ENVIRONMENT}" "${JWT_REFRESH_SECRET}" "JWT refresh token secret"

# Gemini API Key
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${YELLOW}Please enter your Gemini API key (or press Enter to skip):${NC}"
    read -s GEMINI_API_KEY
    if [ ! -z "$GEMINI_API_KEY" ]; then
        create_secret "gemini-api-key" "${GEMINI_API_KEY}" "Google Gemini API key"
    fi
else
    create_secret "gemini-api-key" "${GEMINI_API_KEY}" "Google Gemini API key"
fi

# HuggingFace Token (optional)
if [ ! -z "$HF_TOKEN" ]; then
    create_secret "hf-token" "${HF_TOKEN}" "HuggingFace API token"
fi

# Grant Cloud Run service account access to secrets
echo -e "${YELLOW}Granting Cloud Run service account access to secrets...${NC}"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
CLOUD_RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding jwt-access-secret-${ENVIRONMENT} \
    --member="serviceAccount:${CLOUD_RUN_SA}" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding jwt-refresh-secret-${ENVIRONMENT} \
    --member="serviceAccount:${CLOUD_RUN_SA}" \
    --role="roles/secretmanager.secretAccessor"

if gcloud secrets describe gemini-api-key &>/dev/null; then
    gcloud secrets add-iam-policy-binding gemini-api-key \
        --member="serviceAccount:${CLOUD_RUN_SA}" \
        --role="roles/secretmanager.secretAccessor"
fi

echo -e "${GREEN}Secret Manager setup complete!${NC}"
echo -e "${YELLOW}Secrets created:${NC}"
echo -e "  - jwt-access-secret-${ENVIRONMENT}"
echo -e "  - jwt-refresh-secret-${ENVIRONMENT}"
if gcloud secrets describe gemini-api-key &>/dev/null; then
    echo -e "  - gemini-api-key"
fi

