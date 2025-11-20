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
python -c "import services.scribe_agent.main" 2>&1
IMPORT_EXIT_CODE=$?
if [ $IMPORT_EXIT_CODE -ne 0 ]; then
  echo "ERROR: Cannot import services.scribe_agent.main (exit code: $IMPORT_EXIT_CODE)"
  echo "Python path:"
  python -c "import sys; print('\n'.join(sys.path))"
  echo "Trying to import individual components..."
  python -c "import services" 2>&1 || true
  python -c "import services.scribe_agent" 2>&1 || true
  python -c "from services.scribe_agent import config" 2>&1 || true
  python -c "from shared.config import get_settings" 2>&1 || true
  exit 1
fi

echo "Application module imported successfully"

# Start uvicorn - it will show errors if app creation fails
echo "Starting uvicorn server on port $PORT..."
echo "Note: If app creation fails, uvicorn will show the error"
exec uvicorn services.scribe_agent.main:app --host 0.0.0.0 --port "$PORT" --log-level info --timeout-keep-alive 30 --no-access-log

