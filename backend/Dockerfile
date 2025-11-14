FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_SIZE=base \
    WHISPER_CACHE_DIR=/root/.cache/whisper

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Preload Whisper weights so first request on Cloud Run is fast.
RUN mkdir -p ${WHISPER_CACHE_DIR} \
    && python -c "import os, whisper; whisper.load_model(os.environ.get('MODEL_SIZE', 'base'), download_root=os.environ.get('WHISPER_CACHE_DIR', '/root/.cache/whisper'))"

# Copy the entire backend directory structure
COPY . /app/backend

# Initialize models if they don't exist (for CI builds without local files)
WORKDIR /app/backend
RUN python init_models.py && cp *.pkl /app/

# Use PORT environment variable from Cloud Run, default to 8080 if not set
WORKDIR /app
ENV PORT=8080
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT

