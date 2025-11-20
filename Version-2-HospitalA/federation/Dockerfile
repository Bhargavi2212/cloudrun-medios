FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY federation/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY federation ./federation

# Strict verification - database module MUST be present
RUN echo "=== Verifying database module ===" && \
    ls -la ./database || (echo "ERROR: database directory not found!" && exit 1) && \
    test -f ./database/__init__.py || (echo "ERROR: database/__init__.py not found!" && exit 1) && \
    test -f ./database/session.py || (echo "ERROR: database/session.py not found!" && exit 1) && \
    python -c "import database; print('✓ Database module found')" || \
    (echo "ERROR: Database module import failed!" && \
     echo "Python path:" && python -c "import sys; print('\n'.join(sys.path))" && \
     echo "Current directory contents:" && ls -la . && \
     echo "Database directory contents:" && ls -la ./database && \
     exit 1)

# Copy and make entrypoint executable
COPY federation/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Final runtime verification - ensure database module is accessible at runtime
RUN echo "=== Runtime Verification ===" && \
    python -c "import sys; sys.path.insert(0, '/app'); import database; print('✓ Runtime check passed')" || \
    (echo "ERROR: Runtime verification failed!" && exit 1)

EXPOSE 8010

# Cloud Run sets PORT environment variable
ENTRYPOINT ["./entrypoint.sh"]

