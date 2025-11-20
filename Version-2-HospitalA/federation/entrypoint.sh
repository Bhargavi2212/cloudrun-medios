#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8010
PORT=${PORT:-8010}

exec uvicorn federation.aggregator.main:app --host 0.0.0.0 --port "$PORT"

