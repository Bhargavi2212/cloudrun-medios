FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY services/manage_agent/requirements.txt ./requirements.txt

# Verify requirements.txt includes ML dependencies before installing
RUN echo "=== Verifying requirements.txt ===" && \
    cat requirements.txt && \
    grep -q "joblib" requirements.txt && \
    grep -q "numpy" requirements.txt && \
    grep -q "pandas" requirements.txt && \
    grep -q "scikit-learn" requirements.txt || \
    (echo "ERROR: ML dependencies missing from requirements.txt!" && exit 1)

# Install dependencies with verbose output
RUN echo "=== Installing dependencies ===" && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "=== Verifying installation ===" && \
    pip list | grep -E "joblib|numpy|pandas|scikit" && \
    python -c "import joblib; print('✓ joblib:', joblib.__version__)" && \
    python -c "import numpy; print('✓ numpy:', numpy.__version__)" && \
    python -c "import pandas; print('✓ pandas:', pandas.__version__)" && \
    python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" && \
    echo "✓ All ML dependencies installed successfully"

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

