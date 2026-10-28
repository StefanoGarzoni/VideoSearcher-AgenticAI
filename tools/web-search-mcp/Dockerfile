# Dockerfile for the Web Search MCP Server (StreamableHTTP)
# Lightweight: no system media deps required — pure HTTP search providers.

FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY web_search_main.py .
COPY web_search_server.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8010/health', timeout=5)" || exit 1

# Environment variables
ENV PORT=8010
ENV HOST=0.0.0.0
ENV OUTPUT_DIRECTORY=/data/web_search

EXPOSE 8010

CMD ["python", "web_search_server.py"]
