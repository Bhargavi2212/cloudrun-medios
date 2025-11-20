#!/bin/bash

# Setup free PostgreSQL on Compute Engine (e2-micro - free tier eligible)
# Usage: ./scripts/setup-free-postgres.sh [project-id]

set -e

PROJECT_ID=${1:-${GCP_PROJECT_ID}}
REGION=${GCP_REGION:-us-central1}
ZONE=${GCP_ZONE:-us-central1-a}
INSTANCE_NAME="medios-postgres-free"
DB_USER="medios_user"
DB_NAME="medios_db"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up free PostgreSQL on Compute Engine (e2-micro)...${NC}"
echo -e "Project ID: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo -e "Zone: ${ZONE}"
echo -e "Instance Name: ${INSTANCE_NAME}"
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set. Please provide project ID as argument.${NC}"
    exit 1
fi

# Set the GCP project (skip if already set)
# gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable compute.googleapis.com --project=${PROJECT_ID}

# Generate random password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo -e "${YELLOW}Generated database password: ${DB_PASSWORD}${NC}"
echo -e "${YELLOW}Save this password securely!${NC}"

# Store password in Secret Manager
echo -e "${YELLOW}Storing password in Secret Manager...${NC}"
echo -n "${DB_PASSWORD}" | gcloud secrets create db-password-free \
    --data-file=- \
    --replication-policy="automatic" \
    --project=${PROJECT_ID} 2>/dev/null || \
echo -n "${DB_PASSWORD}" | gcloud secrets versions add db-password-free \
    --data-file=- \
    --project=${PROJECT_ID}

# Check if instance already exists
if gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID} &>/dev/null; then
    echo -e "${YELLOW}Instance ${INSTANCE_NAME} already exists.${NC}"
    INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID} --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
    echo -e "${GREEN}Instance IP: ${INSTANCE_IP}${NC}"
else
    # Create startup script for PostgreSQL installation
    cat > /tmp/postgres-setup.sh << 'EOF'
#!/bin/bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Configure PostgreSQL
sudo -u postgres psql -c "CREATE USER medios_user WITH PASSWORD '${DB_PASSWORD}';"
sudo -u postgres psql -c "CREATE DATABASE medios_db OWNER medios_user;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE medios_db TO medios_user;"

# Configure PostgreSQL to accept connections
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/*/main/postgresql.conf
sudo sed -i "s/#port = 5432/port = 5432/" /etc/postgresql/*/main/postgresql.conf

# Configure pg_hba.conf to allow connections
echo "host    all             all             0.0.0.0/0               md5" | sudo tee -a /etc/postgresql/*/main/pg_hba.conf

# Restart PostgreSQL
sudo systemctl restart postgresql
sudo systemctl enable postgresql

# Configure firewall (if ufw is installed)
sudo ufw allow 5432/tcp || true
EOF

    # Replace password in startup script
    sed -i "s/\${DB_PASSWORD}/${DB_PASSWORD}/g" /tmp/postgres-setup.sh

    # Create e2-micro instance (free tier eligible in us-central1, us-east1, us-west1)
    echo -e "${YELLOW}Creating e2-micro instance (free tier eligible)...${NC}"
    gcloud compute instances create ${INSTANCE_NAME} \
        --zone=${ZONE} \
        --machine-type=e2-micro \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=30GB \
        --boot-disk-type=pd-standard \
        --tags=postgres-server \
        --metadata-from-file startup-script=/tmp/postgres-setup.sh \
        --project=${PROJECT_ID}

    echo -e "${YELLOW}Waiting for instance to start and PostgreSQL to be configured...${NC}"
    sleep 30

    # Get instance IP
    INSTANCE_IP=$(gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --project=${PROJECT_ID} --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
    echo -e "${GREEN}Instance created! IP: ${INSTANCE_IP}${NC}"

    # Create firewall rule to allow PostgreSQL connections from Cloud Run
    echo -e "${YELLOW}Creating firewall rule...${NC}"
    gcloud compute firewall-rules create allow-postgres-cloudrun \
        --allow tcp:5432 \
        --source-ranges 0.0.0.0/0 \
        --target-tags postgres-server \
        --description "Allow PostgreSQL from Cloud Run" \
        --project=${PROJECT_ID} 2>/dev/null || echo "Firewall rule may already exist"
fi

# Wait a bit more for PostgreSQL to be fully ready
echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
sleep 20

# Test connection (optional)
echo -e "${YELLOW}Testing connection...${NC}"
PGPASSWORD=${DB_PASSWORD} psql -h ${INSTANCE_IP} -U ${DB_USER} -d ${DB_NAME} -c "SELECT version();" 2>/dev/null && \
    echo -e "${GREEN}Connection successful!${NC}" || \
    echo -e "${YELLOW}Connection test skipped (psql client may not be installed locally)${NC}"

echo -e "${GREEN}PostgreSQL setup complete!${NC}"
echo -e "Instance Name: ${INSTANCE_NAME}"
echo -e "Instance IP: ${INSTANCE_IP}"
echo -e "Database: ${DB_NAME}"
echo -e "User: ${DB_USER}"
echo -e "Password: ${DB_PASSWORD} (stored in Secret Manager as db-password-free)"
echo ""
echo -e "${YELLOW}Connection string for Cloud Run:${NC}"
echo -e "postgresql+asyncpg://${DB_USER}:${DB_PASSWORD}@${INSTANCE_IP}:5432/${DB_NAME}"
echo ""
echo -e "${YELLOW}Note: This uses e2-micro which is free tier eligible in us-central1, us-east1, us-west1${NC}"
echo -e "${YELLOW}Make sure to update your CI/CD workflow with this connection string!${NC}"

