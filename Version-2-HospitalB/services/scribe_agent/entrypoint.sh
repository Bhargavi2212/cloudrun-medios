#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8002
PORT=${PORT:-8002}

echo "Starting scribe-agent on port $PORT..."

# Start uvicorn with error handling
exec uvicorn services.scribe_agent.main:app --host 0.0.0.0 --port "$PORT" --log-level info

