from nvidia/cuda:12.8.0-devel-ubuntu22.04

env DEBIAN_FRONTEND=noninteractive
env PYTHONUNBUFFERED=1
env PYTHONDONTWRITEBYTECODE=1
env HF_HOME=/root/.cache/huggingface
env HF_HUB_ENABLE_HF_TRANSFER=1

run apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

run ln -sf /usr/bin/python3.12 /usr/bin/python
run python3.12 -m ensurepip --upgrade
run python -m pip install --upgrade pip

workdir /app

copy requirements.txt ./
run --mount=type=cache,target=/root/.cache/pip \
    PIP_CACHE_DIR=/root/.cache/pip pip install -r requirements.txt

copy apis.py apis.py
expose 8000

cmd ["uvicorn", "apis:app", "--host", "0.0.0.0", "--port", "8000"]