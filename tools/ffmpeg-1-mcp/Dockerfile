FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ffmpeg_1_server.py .

# Create output directory
RUN mkdir -p /tmp/ffmpeg_output

# Expose port for StreamableHTTP
EXPOSE 8002

# Run the server
CMD ["python", "ffmpeg_1_server.py"]
