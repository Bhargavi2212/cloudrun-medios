#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 9003
PORT=${PORT:-9003}

exec uvicorn services.summarizer_agent.main:app --host 0.0.0.0 --port "$PORT"

