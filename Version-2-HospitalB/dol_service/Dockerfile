FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY dol_service/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY shared ./shared
COPY database ./database
COPY dol_service ./dol_service

EXPOSE 8004

CMD ["uvicorn", "dol_service.main:app", "--host", "0.0.0.0", "--port", "8004"]

