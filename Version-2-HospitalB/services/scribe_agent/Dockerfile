FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY services/scribe_agent/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY services/scribe_agent ./services/scribe_agent

EXPOSE 9002

CMD ["sh", "-c", "uvicorn services.scribe_agent.main:app --host 0.0.0.0 --port ${PORT:-9002}"]

