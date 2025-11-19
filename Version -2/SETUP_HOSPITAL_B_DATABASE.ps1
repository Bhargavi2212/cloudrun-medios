# Setup Hospital B Database
# Creates and initializes a new database for Hospital B (County Hospital)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Hospital B Database" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Database connection details
$dbHost = "localhost"
$dbPort = "5432"
$dbUser = "postgres"
$dbPassword = "Anuradha"
$dbName = "medi_os_v2_hospital_b"

Write-Host "Step 1: Creating database '$dbName'..." -ForegroundColor Yellow

# Set PGPASSWORD environment variable for psql
$env:PGPASSWORD = $dbPassword

# Create database (ignore error if it already exists)
$createDbQuery = "SELECT 1 FROM pg_database WHERE datname = '$dbName';"
$dbExists = psql -h $dbHost -p $dbPort -U $dbUser -d postgres -tAc $createDbQuery

if ($dbExists -eq "1") {
    Write-Host "Database '$dbName' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to drop and recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-Host "Dropping existing database..." -ForegroundColor Yellow
        psql -h $dbHost -p $dbPort -U $dbUser -d postgres -c "DROP DATABASE $dbName;"
        psql -h $dbHost -p $dbPort -U $dbUser -d postgres -c "CREATE DATABASE $dbName;"
        Write-Host "Database recreated successfully!" -ForegroundColor Green
    } else {
        Write-Host "Using existing database." -ForegroundColor Green
    }
} else {
    psql -h $dbHost -p $dbPort -U $dbUser -d postgres -c "CREATE DATABASE $dbName;"
    Write-Host "Database created successfully!" -ForegroundColor Green
}

Write-Host ""
Write-Host "Step 2: Initializing database schema..." -ForegroundColor Yellow

# Set database URL for migrations
$dbUrl = "postgresql+asyncpg://$dbUser`:$dbPassword@$dbHost`:$dbPort/$dbName"
$env:DATABASE_URL = $dbUrl

# Run Alembic migrations
Write-Host "Running database migrations..." -ForegroundColor Cyan
cd "D:\Hackathons\Cloud Run\Version -2"
poetry run alembic -c database/alembic.ini upgrade head

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Hospital B Database Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Database: $dbName" -ForegroundColor White
    Write-Host "Connection: $dbUrl" -ForegroundColor Gray
    Write-Host ""
    Write-Host "You can now start Hospital B services using:" -ForegroundColor Yellow
    Write-Host "  .\START_HOSPITAL_B.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Migration failed. Please check the error above." -ForegroundColor Red
    Write-Host "You may need to initialize the schema manually." -ForegroundColor Yellow
}

# Clear password from environment
Remove-Item Env:\PGPASSWORD
