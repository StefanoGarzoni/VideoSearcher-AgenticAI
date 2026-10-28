# Dockerfile for Transcription MCP Server with StreamableHTTP
# Cloud-native deployment of audio transcription microservice

# Use standard Python image (for CPU inference)
# For GPU support, use: nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip (force overwrite if exists)
#RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
#    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip and install build tools
#RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and entrypoint
COPY transcribe_main.py .
COPY transcribe_server.py .
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh && \
    ls -la /app/entrypoint.sh

# Create output directory
RUN mkdir -p /tmp/ffmpeg_output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8004/health', timeout=5)" || exit 1

# Environment variables
ENV PORT=8004
ENV HOST=0.0.0.0
ENV OUTPUT_DIRECTORY=/tmp/ffmpeg_output
ENV ENABLE_SESSION_MANAGEMENT=true
ENV HF_HOME=/app/.cache/huggingface

# Expose port
EXPOSE 8004

# Use entrypoint to handle model download
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

# Run the server
CMD ["python", "transcribe_server.py"]