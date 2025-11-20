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

# Create data directory structure for model files
# Models will use fallback heuristics if files are not present
RUN mkdir -p ./data/receptionist_models_v3/models \
    ./data/nurse_models/models \
    ./data/processed/transformers

EXPOSE 8001

CMD ["uvicorn", "services.manage_agent.main:app", "--host", "0.0.0.0", "--port", "8001"]

