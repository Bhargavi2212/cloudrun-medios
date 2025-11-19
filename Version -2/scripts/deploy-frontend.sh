#!/bin/bash

# Deploy Medi OS Frontend to Google Cloud Run
# Usage: ./scripts/deploy-frontend.sh [environment] [project-id] [region]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${3:-us-central1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying Medi OS Frontend to Cloud Run...${NC}"
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
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet

# Build frontend
echo -e "${YELLOW}Building frontend...${NC}"
cd apps/frontend

# Install dependencies
npm install

# Build with production environment variables
# Get backend URLs (they should be deployed first)
MANAGE_URL="https://manage-agent-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
SCRIBE_URL="https://scribe-agent-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
SUMMARIZER_URL="https://summarizer-agent-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
DOL_URL="https://dol-service-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"
FEDERATION_URL="https://federation-${ENVIRONMENT}-${REGION}-${PROJECT_ID}.a.run.app"

VITE_MANAGE_API_URL=${MANAGE_URL} \
VITE_SCRIBE_API_URL=${SCRIBE_URL} \
VITE_SUMMARIZER_API_URL=${SUMMARIZER_URL} \
VITE_DOL_API_URL=${DOL_URL} \
VITE_FEDERATION_API_URL=${FEDERATION_URL} \
npm run build

cd ../..

# Create a simple nginx Dockerfile for the frontend
cat > apps/frontend/Dockerfile.prod <<EOF
FROM nginx:alpine
COPY dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
EOF

# Create nginx config for SPA routing
cat > apps/frontend/nginx.conf <<EOF
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # SPA routing - all routes go to index.html
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Build and push Docker image
IMAGE_NAME="gcr.io/${PROJECT_ID}/medios-frontend"
CLOUD_RUN_SERVICE="medios-frontend-${ENVIRONMENT}"

echo -e "${YELLOW}Building Docker image...${NC}"
cd apps/frontend
gcloud builds submit \
    --tag ${IMAGE_NAME}:${ENVIRONMENT} \
    --tag ${IMAGE_NAME}:latest \
    -f Dockerfile.prod \
    .
cd ../..

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying frontend to Cloud Run...${NC}"
gcloud run deploy ${CLOUD_RUN_SERVICE} \
    --image ${IMAGE_NAME}:${ENVIRONMENT} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --cpu 1 \
    --timeout 60 \
    --max-instances 10 \
    --min-instances 0 \
    --concurrency 80 \
    --port 80

# Get the service URL
SERVICE_URL=$(gcloud run services describe ${CLOUD_RUN_SERVICE} --region ${REGION} --format 'value(status.url)')

echo -e "${GREEN}Frontend deployment complete!${NC}"
echo -e "Service URL: ${SERVICE_URL}"

