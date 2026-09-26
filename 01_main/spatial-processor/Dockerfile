FROM python:3.11-slim

WORKDIR /app

COPY spatial-processor/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY spatial-processor/src ./src

ENV PYTHONUNBUFFERED=1
WORKDIR /app/src

CMD ["python", "main.py"]
