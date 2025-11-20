FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY federation/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY federation ./federation

# Copy and make entrypoint executable
COPY federation/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

EXPOSE 8010

# Cloud Run sets PORT environment variable
ENTRYPOINT ["./entrypoint.sh"]

