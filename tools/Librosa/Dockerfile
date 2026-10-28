FROM python:3.12-slim

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify key imports work
RUN python -c "import librosa; print('✓ librosa')"
RUN python -c "import fastapi; print('✓ fastapi')"

COPY api.py .

EXPOSE 8001

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
