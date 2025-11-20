#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8001
PORT=${PORT:-8001}

exec uvicorn services.manage_agent.main:app --host 0.0.0.0 --port "$PORT"

