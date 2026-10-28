# Dockerfile for the Summarize MCP Server (StreamableHTTP)
# Lightweight: needs only tar (present in the base image) + an LLM HTTP client.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_client.py .
COPY summarize_main.py .
COPY summarize_server.py .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8016/health', timeout=5)" || exit 1

ENV PORT=8016
ENV HOST=0.0.0.0
ENV OUTPUT_DIRECTORY=/data/summarization

EXPOSE 8016

CMD ["python", "summarize_server.py"]
