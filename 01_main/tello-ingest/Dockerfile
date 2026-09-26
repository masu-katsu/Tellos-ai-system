FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev \
        libgl1 \
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY tello-ingest/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY tello-ingest/src ./src

ENV PYTHONUNBUFFERED=1
WORKDIR /app/src

CMD ["python", "main.py"]
