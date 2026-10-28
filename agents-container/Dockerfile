# ── Base mirrors VLMs-Testing-Framework/Dockerfile ──────────────────────────
# Update this section whenever VLMs-Testing-Framework/Dockerfile changes.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/usr/local/cuda/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN pip install --no-cache-dir torch==2.4.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR /app

# VLMs-Testing-Framework source from git submodule — needed by VqaAgent
COPY VLMs-Testing-Framework/src /vlm_src

# Build context is repo root — prefix paths with agents-container/
COPY agents-container/requirements.txt /app/requirements.txt
# requirements.txt upgrades torch to >=2.7.1 for Phi-4 and adds VLM deps
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agents-container/ /app

CMD ["python", "videosearcher_cli.py"]
