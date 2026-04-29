# syntax=docker/dockerfile:1.5
FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install faster-whisper fastapi uvicorn python-multipart

WORKDIR /app
COPY server.py .

ENV DATA_DIR=/data

EXPOSE 80
CMD ["python", "server.py"]
