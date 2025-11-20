#!/bin/bash
# Install PostgreSQL
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib

# Configure PostgreSQL
sudo -u postgres psql -c "CREATE USER medios_user WITH PASSWORD 'm4zrH73diqa1yW2EFNkvshZxU';"
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

