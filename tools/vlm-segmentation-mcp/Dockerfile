# ── Base mirrors VLMs-Testing-Framework/Dockerfile ──────────────────────────
# Update this section whenever VLMs-Testing-Framework/Dockerfile changes.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl libgl1 libglib2.0-0 \
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

# VLMs-Testing-Framework source code (from git submodule at repo root)
# Baked into the image — no host volume mount needed, works for anyone cloning the repo.
COPY VLMs-Testing-Framework/src /vlm_src

COPY tools/vlm-segmentation-mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tools/vlm-segmentation-mcp/vlm_segmentation_server.py .

# Bake BOTH segmentation checkpoints into the image at build time so the image is
# fully self-contained — no runtime downloads, no network dependency on first boot.
# On a fresh machine Docker copies these into the (empty) vlm_weights volume from
# this image layer, so /app/weights/ is populated deterministically. The md5 check
# fails the build loudly if either download is truncated/corrupt (a silent bad file
# used to surface only as model_loaded=false and an endless "11/12 servers ready").
RUN mkdir -p /app/weights && \
    wget -q -O /app/weights/groundingdino_swint_ogc.pth \
      https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
    wget -q -O /app/weights/sam2_hiera_tiny.pt \
      https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt && \
    echo "99eacccce4ada0b35153d4fd7af05297  /app/weights/sam2_hiera_tiny.pt" | md5sum -c -

CMD ["python3.11", "-m", "uvicorn", "vlm_segmentation_server:app", "--host", "0.0.0.0", "--port", "8008"]
