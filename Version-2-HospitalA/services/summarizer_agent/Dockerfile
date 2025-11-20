FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY services/summarizer_agent/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY services/summarizer_agent ./services/summarizer_agent

EXPOSE 8003

CMD ["sh", "-c", "uvicorn services.summarizer_agent.main:app --host 0.0.0.0 --port ${PORT:-8003}"]

