FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY federation/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY federation ./federation

# Verify database module is present
RUN python -c "import database; print('✓ Database module found')" || \
    (echo "ERROR: Database module missing!" && ls -la ./database && exit 1)

# Copy and make entrypoint executable
COPY federation/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8010

# Cloud Run sets PORT environment variable
ENTRYPOINT ["./entrypoint.sh"]

