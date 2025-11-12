FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_SIZE=base

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Preload Whisper weights so first request on Cloud Run is fast.
RUN python -c "import os, whisper; whisper.load_model(os.environ.get('MODEL_SIZE', 'base'))"

# Copy the entire backend directory structure
COPY . /app/backend

# Use PORT environment variable from Cloud Run, default to 8080 if not set
ENV PORT=8080
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT

