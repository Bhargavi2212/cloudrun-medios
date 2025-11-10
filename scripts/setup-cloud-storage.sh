#!/bin/bash

# Setup Google Cloud Storage bucket for Medi OS
# Usage: ./scripts/setup-cloud-storage.sh [environment] [project-id]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${GCP_REGION:-us-central1}
BUCKET_NAME="${PROJECT_ID}-medios-storage-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Cloud Storage bucket...${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo -e "Bucket Name: ${BUCKET_NAME}"
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
gcloud services enable storage-api.googleapis.com
gcloud services enable storage-component.googleapis.com

# Check if bucket already exists
if gsutil ls -b gs://${BUCKET_NAME} &>/dev/null; then
    echo -e "${YELLOW}Bucket ${BUCKET_NAME} already exists. Skipping creation.${NC}"
else
    # Create storage bucket
    echo -e "${YELLOW}Creating storage bucket...${NC}"
    gsutil mb -p ${PROJECT_ID} -c STANDARD -l ${REGION} gs://${BUCKET_NAME}
fi

# Set bucket lifecycle policy
echo -e "${YELLOW}Setting up lifecycle policy...${NC}"
cat > /tmp/lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 2555,
          "matchesPrefix": ["documents/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 365,
          "matchesPrefix": ["audio/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json gs://${BUCKET_NAME}
rm /tmp/lifecycle.json

# Set CORS policy for frontend access
echo -e "${YELLOW}Setting up CORS policy...${NC}"
cat > /tmp/cors.json << EOF
[
  {
    "origin": ["https://medios-frontend-*.run.app", "http://localhost:3000"],
    "method": ["GET", "POST", "PUT", "DELETE", "HEAD"],
    "responseHeader": ["Content-Type", "Authorization"],
    "maxAgeSeconds": 3600
  }
]
EOF

gsutil cors set /tmp/cors.json gs://${BUCKET_NAME}
rm /tmp/cors.json

# Set uniform bucket-level access
echo -e "${YELLOW}Configuring bucket access...${NC}"
gsutil uniformbucketlevelaccess set on gs://${BUCKET_NAME}

# Grant Cloud Run service account access
echo -e "${YELLOW}Granting Cloud Run service account access...${NC}"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
CLOUD_RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gsutil iam ch serviceAccount:${CLOUD_RUN_SA}:objectAdmin gs://${BUCKET_NAME}

# Create folder structure
echo -e "${YELLOW}Creating folder structure...${NC}"
gsutil mkdir gs://${BUCKET_NAME}/audio
gsutil mkdir gs://${BUCKET_NAME}/documents
gsutil mkdir gs://${BUCKET_NAME}/uploads

echo -e "${GREEN}Cloud Storage setup complete!${NC}"
echo -e "Bucket Name: ${BUCKET_NAME}"
echo -e "Region: ${REGION}"
echo -e "URL: gs://${BUCKET_NAME}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "STORAGE_BACKEND=gcs"
echo -e "STORAGE_GCS_BUCKET=${BUCKET_NAME}"

