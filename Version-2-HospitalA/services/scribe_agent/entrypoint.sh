#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8002
PORT=${PORT:-8002}

echo "Starting scribe-agent on port $PORT..."
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Uvicorn path: $(which uvicorn)"

# Verify the application module exists
if ! python -c "import services.scribe_agent.main" 2>/dev/null; then
  echo "ERROR: Cannot import services.scribe_agent.main"
  python -c "import sys; print('Python path:', sys.path)"
  exit 1
fi

# Start uvicorn with error handling
exec uvicorn services.scribe_agent.main:app --host 0.0.0.0 --port "$PORT" --log-level info

