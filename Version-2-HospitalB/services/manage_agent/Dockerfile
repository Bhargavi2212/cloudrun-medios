FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY services/manage_agent/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY dol_service ./dol_service
COPY services/manage_agent ./services/manage_agent

# Create data directory structure for model files
# Models will use fallback heuristics if files are not present
RUN mkdir -p ./data/receptionist_models_v3/models \
    ./data/nurse_models/models \
    ./data/processed/transformers

# Verify all ML dependencies are installed (force rebuild if missing)
RUN python -c "import joblib, numpy, pandas, sklearn; print('✓ All ML dependencies available')" || \
    (echo "ERROR: ML dependencies missing!" && pip list | grep -E "joblib|numpy|pandas|scikit" && exit 1)

# Copy and make entrypoint executable
COPY services/manage_agent/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8001

# Cloud Run sets PORT environment variable
ENTRYPOINT ["./entrypoint.sh"]

