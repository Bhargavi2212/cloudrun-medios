FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY services/manage_agent/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY services/manage_agent ./services/manage_agent

EXPOSE 9001

CMD ["sh", "-c", "uvicorn services.manage_agent.main:app --host 0.0.0.0 --port ${PORT:-9001}"]

