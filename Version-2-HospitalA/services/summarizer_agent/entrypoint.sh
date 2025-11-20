#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8003
PORT=${PORT:-8003}

exec uvicorn services.summarizer_agent.main:app --host 0.0.0.0 --port "$PORT"

