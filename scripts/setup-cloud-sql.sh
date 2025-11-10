#!/bin/bash

# Setup Google Cloud SQL PostgreSQL instance for Medi OS
# Usage: ./scripts/setup-cloud-sql.sh [environment] [project-id]

set -e

ENVIRONMENT=${1:-production}
PROJECT_ID=${2:-${GCP_PROJECT_ID}}
REGION=${GCP_REGION:-us-central1}
INSTANCE_NAME="medios-db-${ENVIRONMENT}"
DATABASE_NAME="medios_db"
USER_NAME="medios_user"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Cloud SQL PostgreSQL instance...${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo -e "Instance Name: ${INSTANCE_NAME}"
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
gcloud services enable sqladmin.googleapis.com
gcloud services enable compute.googleapis.com

# Check if instance already exists
if gcloud sql instances describe ${INSTANCE_NAME} &>/dev/null; then
    echo -e "${YELLOW}Instance ${INSTANCE_NAME} already exists. Skipping creation.${NC}"
else
    # Create Cloud SQL instance
    echo -e "${YELLOW}Creating Cloud SQL instance...${NC}"
    gcloud sql instances create ${INSTANCE_NAME} \
        --database-version=POSTGRES_15 \
        --tier=db-f1-micro \
        --region=${REGION} \
        --storage-type=SSD \
        --storage-size=10GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --enable-bin-log \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=04 \
        --maintenance-release-channel=production \
        --deletion-protection
fi

# Get instance connection name
CONNECTION_NAME=$(gcloud sql instances describe ${INSTANCE_NAME} --format="value(connectionName)")

# Generate random password if not provided
if [ -z "$DB_PASSWORD" ]; then
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    echo -e "${YELLOW}Generated database password: ${DB_PASSWORD}${NC}"
    echo -e "${YELLOW}Save this password securely!${NC}"
fi

# Create database
echo -e "${YELLOW}Creating database...${NC}"
gcloud sql databases create ${DATABASE_NAME} \
    --instance=${INSTANCE_NAME} \
    --charset=utf8 \
    --collation=utf8_general_ci || echo "Database may already exist"

# Create user
echo -e "${YELLOW}Creating database user...${NC}"
gcloud sql users create ${USER_NAME} \
    --instance=${INSTANCE_NAME} \
    --password=${DB_PASSWORD} || echo "User may already exist"

# Store password in Secret Manager
echo -e "${YELLOW}Storing password in Secret Manager...${NC}"
echo -n "${DB_PASSWORD}" | gcloud secrets create db-password-${ENVIRONMENT} \
    --data-file=- \
    --replication-policy="automatic" \
    --project=${PROJECT_ID} 2>/dev/null || \
echo -n "${DB_PASSWORD}" | gcloud secrets versions add db-password-${ENVIRONMENT} \
    --data-file=- \
    --project=${PROJECT_ID}

# Grant Cloud Run service account access to the instance
echo -e "${YELLOW}Configuring Cloud Run access...${NC}"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
CLOUD_RUN_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${CLOUD_RUN_SA}" \
    --role="roles/cloudsql.client"

# Create authorized network (optional, for local access)
echo -e "${YELLOW}To allow local access, add your IP to authorized networks:${NC}"
echo -e "gcloud sql instances patch ${INSTANCE_NAME} --authorized-networks=$(curl -s ifconfig.me)/32"

echo -e "${GREEN}Cloud SQL setup complete!${NC}"
echo -e "Connection Name: ${CONNECTION_NAME}"
echo -e "Database: ${DATABASE_NAME}"
echo -e "User: ${USER_NAME}"
echo -e "Password: ${DB_PASSWORD} (stored in Secret Manager)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Run database migrations: alembic upgrade head"
echo -e "2. Update Cloud Run service to use this database"
echo -e "3. Connection string: postgresql+psycopg2://${USER_NAME}:${DB_PASSWORD}@/medios_db?host=/cloudsql/${CONNECTION_NAME}"

