# Manage Agent Service Failure - Root Causes and Fixes

## Issues Found

### 1. ✅ Import Error (FIXED)
**Error**: `ImportError: cannot import name 'get_federated_sync_service' from 'services.manage_agent.dependencies'`

**Root Cause**: The `dol_service` directory was not being copied into the Docker container, causing import failures when `federated_sync_service.py` tried to import from `dol_service`.

**Fix Applied**: Added `COPY dol_service ./dol_service` to `services/manage_agent/Dockerfile`

**Status**: Fixed in code, needs redeployment

---

### 2. ✅ Database Connection Error (FIXED)
**Error**: `relation "patients" does not exist` and database connection issues

**Root Cause**: 
- The deployment script was using Cloud SQL connection format (`host=/cloudsql/...`) 
- Your database is on a Compute Engine VM, not Cloud SQL
- The DATABASE_URL was hardcoded with incorrect format and placeholder credentials

**Fix Applied**: Updated `scripts/deploy-service.sh` to:
- Get DATABASE_URL from Secret Manager (secret name: `database-url-${ENVIRONMENT}`)
- Fallback to constructing from individual secrets (`db-host`, `db-user`, `db-password`, `db-name`, `db-port`)
- Use proper PostgreSQL connection string format for VM: `postgresql+asyncpg://user:password@host:port/database`
- Removed Cloud SQL connection setup

**Status**: Fixed in code, needs configuration

---

## Next Steps

### 1. Set Up Database URL Secret

You need to create a secret in Google Cloud Secret Manager with your VM database connection details:

```bash
# Option 1: Store complete DATABASE_URL as a single secret (recommended)
gcloud secrets create database-url-production \
    --data-file=- <<< "postgresql+asyncpg://username:password@vm-ip-or-hostname:5432/medios_db" \
    --project=plasma-datum-476821-s8

# Option 2: Store individual components
gcloud secrets create db-host-production --data-file=- <<< "your-vm-ip-or-hostname"
gcloud secrets create db-user-production --data-file=- <<< "postgres"
gcloud secrets create db-password-production --data-file=- <<< "your-db-password"
gcloud secrets create db-name-production --data-file=- <<< "medios_db"
gcloud secrets create db-port-production --data-file=- <<< "5432"
```

**Important**: Replace:
- `username` with your PostgreSQL username
- `password` with your PostgreSQL password  
- `vm-ip-or-hostname` with your VM's internal IP or hostname
- `medios_db` with your actual database name

### 2. Ensure Database Schema is Created

The error `relation "patients" does not exist` means the database tables haven't been created. You need to run migrations:

```bash
# Connect to your VM database and run migrations
# Option 1: From your local machine (if you have access)
cd "Version -2"
export DATABASE_URL="postgresql+asyncpg://user:password@vm-ip:5432/medios_db"
alembic upgrade head

# Option 2: Run migrations from Cloud Run Job
# (Create a Cloud Run job that runs alembic upgrade head)
```

### 3. Ensure VM Database is Accessible from Cloud Run

For Cloud Run to connect to your VM database, you need:

1. **VM Firewall Rules**: Allow incoming connections on port 5432 from Cloud Run
   ```bash
   # Get Cloud Run IP ranges (if using VPC connector) or allow all Cloud Run IPs
   gcloud compute firewall-rules create allow-cloud-run-postgres \
       --allow tcp:5432 \
       --source-ranges 0.0.0.0/0 \
       --target-tags postgres-server
   ```

2. **VPC Connector (Recommended)**: Set up a VPC connector for Cloud Run to access VM
   ```bash
   # Create VPC connector
   gcloud compute networks vpc-access connectors create cloud-run-connector \
       --region=us-central1 \
       --subnet=default \
       --subnet-project=plasma-datum-476821-s8
   ```

3. **PostgreSQL Configuration**: Ensure PostgreSQL on VM allows connections:
   - Update `postgresql.conf`: `listen_addresses = '*'` or specific IPs
   - Update `pg_hba.conf`: Allow connections from Cloud Run IP ranges

### 4. Redeploy Services

After setting up the secrets and database:

```bash
cd "Version -2"
./scripts/deploy-service.sh manage-agent production plasma-datum-476821-s8 us-central1
```

---

## Verification

### Check Service Status
```bash
gcloud run services list --filter="name:medi-os-hospital-*-manage-agent"
```

### Check Logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=medi-os-hospital-a-manage-agent" --limit 20
```

### Test Database Connection
```bash
# Check if service can connect to database
gcloud run services describe medi-os-hospital-a-manage-agent --region us-central1 --format="value(status.url)"
curl https://your-service-url/health
```

---

## Summary

✅ **Import Error**: Fixed by adding `dol_service` to Dockerfile  
✅ **Database Connection**: Fixed by updating deployment script to use VM connection format  
⏳ **Next**: Set up DATABASE_URL secret and ensure database schema exists  
⏳ **Next**: Ensure network connectivity between Cloud Run and VM  

Once secrets are configured and database is accessible, redeploy the services.

