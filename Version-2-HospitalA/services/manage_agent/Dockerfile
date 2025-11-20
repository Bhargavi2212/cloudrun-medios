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

# Install dependencies with verbose output and strict verification
RUN echo "=== Installing dependencies ===" && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "=== Verifying installation ===" && \
    pip list | grep -E "joblib|numpy|pandas|scikit" || (echo "ERROR: Dependencies not found in pip list!" && exit 1) && \
    python -c "import joblib; print('✓ joblib:', joblib.__version__)" || (echo "ERROR: joblib import failed!" && exit 1) && \
    python -c "import numpy; print('✓ numpy:', numpy.__version__)" || (echo "ERROR: numpy import failed!" && exit 1) && \
    python -c "import pandas; print('✓ pandas:', pandas.__version__)" || (echo "ERROR: pandas import failed!" && exit 1) && \
    python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" || (echo "ERROR: sklearn import failed!" && exit 1) && \
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
# Copy model files if they exist (optional - models will use fallback if missing)
# Note: Model files are too large for git, so they may not be in build context
# They can be copied from Version -2/data during CI/CD build if needed
COPY data/nurse_models ./data/nurse_models
COPY data/processed/transformers ./data/processed/transformers

# Final verification - MUST pass or build fails
RUN python -c "import joblib, numpy, pandas, sklearn; print('✓ All ML dependencies available')" || \
    (echo "ERROR: ML dependencies missing in final verification!" && \
     echo "Checking pip list:" && pip list | grep -E "joblib|numpy|pandas|scikit" || echo "No ML packages found!" && \
     echo "Python path:" && python -c "import sys; print('\n'.join(sys.path))" && \
     exit 1)

# Copy and make entrypoint executable
COPY services/manage_agent/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Final runtime verification - ensure dependencies are accessible at runtime
RUN echo "=== Runtime Verification ===" && \
    python -c "import sys; sys.path.insert(0, '/app'); import joblib, numpy, pandas, sklearn; print('✓ Runtime check passed')" || \
    (echo "ERROR: Runtime verification failed!" && exit 1)

EXPOSE 8001

# Cloud Run sets PORT environment variable
ENTRYPOINT ["./entrypoint.sh"]

