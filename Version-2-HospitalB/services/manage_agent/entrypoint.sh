#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 9001
PORT=${PORT:-9001}

exec uvicorn services.manage_agent.main:app --host 0.0.0.0 --port "$PORT"

