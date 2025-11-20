#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8004
PORT=${PORT:-8004}

exec uvicorn dol_service.main:app --host 0.0.0.0 --port "$PORT"

