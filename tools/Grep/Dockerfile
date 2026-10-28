# Dockerfile for FFmpeg MCP Server with StreamableHTTP
# Cloud-native deployment of audio extraction microservice

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY grep_main.py .
COPY grep_server.py .

# Create output directory
RUN mkdir -p /tmp/ffmpeg_output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8005/health', timeout=5)" || exit 1

# Environment variables
ENV PORT=8005
ENV HOST=0.0.0.0
ENV OUTPUT_DIRECTORY=/tmp/ffmpeg_output
ENV ENABLE_SESSION_MANAGEMENT=true

# Expose port
EXPOSE 8005

# Run the server
CMD ["python", "grep_server.py"]