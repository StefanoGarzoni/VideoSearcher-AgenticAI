# Dockerfile for the Scene Detect MCP Server (StreamableHTTP)
# Requires ffmpeg for PySceneDetect video backend support.

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scene_detect_main.py .
COPY scene_detect_server.py .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8025/health', timeout=5)" || exit 1

ENV PORT=8025
ENV HOST=0.0.0.0
ENV OUTPUT_DIRECTORY=/data/scenes

EXPOSE 8025

CMD ["python", "scene_detect_server.py"]
