#!/bin/sh
set -e

# Use PORT environment variable if set, otherwise default to 8002
PORT=${PORT:-8002}

echo "Starting scribe-agent on port $PORT..."
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Uvicorn path: $(which uvicorn)"
echo "Python version: $(python --version)"

# List directory structure
echo "Directory contents:"
ls -la /app || true
echo "Services directory:"
ls -la /app/services || true
echo "Scribe agent directory:"
ls -la /app/services/scribe_agent || true

# Verify the application module exists - show actual errors
echo "Checking if application module can be imported..."
if ! python -c "import services.scribe_agent.main" 2>&1; then
  echo "ERROR: Cannot import services.scribe_agent.main"
  echo "Python path:"
  python -c "import sys; print('\n'.join(sys.path))"
  exit 1
fi

echo "Application module imported successfully"

# Start uvicorn with error handling
echo "Starting uvicorn server..."
exec uvicorn services.scribe_agent.main:app --host 0.0.0.0 --port "$PORT" --log-level info

