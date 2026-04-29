FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir edge-tts fastapi uvicorn

WORKDIR /app
COPY server.py .

EXPOSE 8881
CMD ["python", "server.py"]
