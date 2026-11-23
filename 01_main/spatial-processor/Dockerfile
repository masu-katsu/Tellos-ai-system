FROM python:3.11-slim

WORKDIR /app

COPY spatial-processor/requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY common ./common
COPY spatial-processor/src ./src

ENV PYTHONUNBUFFERED=1
WORKDIR /app/src

CMD ["python", "main.py"]
