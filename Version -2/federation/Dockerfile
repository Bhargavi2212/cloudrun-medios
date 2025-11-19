FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY federation/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY federation ./federation

EXPOSE 8010

CMD ["uvicorn", "federation.aggregator.main:app", "--host", "0.0.0.0", "--port", "8010"]

