FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git wget curl unzip \
    build-essential cmake ninja-build \
    libgl1 libglib2.0-0 \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ============================================================
# Heavy Python dependencies (baked into image — rarely change)
# ============================================================
# PyTorch 2.9.1 + CUDA 13.0 — pin the FULL matched trio (torch/vision/audio move
# together; an unpinned torchaudio drifts out of ABI lockstep -> AI-Toolkit's
# `import torchaudio` fails with `undefined symbol: torch_library_impl`).
RUN pip install --no-cache-dir --break-system-packages \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

RUN pip install --no-cache-dir --break-system-packages \
    safetensors>=0.4.0 \
    accelerate>=1.2.1 \
    pyyaml>=6.0 \
    runpod>=1.7.0 \
    boto3>=1.35.0 \
    loguru>=0.7.0 \
    toml>=0.10.0 \
    httpx>=0.27.0 \
    requests>=2.31.0 \
    huggingface_hub>=0.34.0 \
    hf_xet>=1.1.0
# NOTE: transformers/diffusers/peft/huggingface_hub omitted here — AI-Toolkit
# requirements.txt pins exact versions (transformers==5.5.3, peft==0.18.1,
# huggingface_hub==1.10.1, diffusers@git-SHA, torchao==0.10.0) and installing
# them twice causes resolver thrash. Let AI-Toolkit's requirements win (§9).

# ============================================================
# Clone AI-Toolkit (replaces diffusion-pipe)
# ============================================================
ARG AI_TOOLKIT_COMMIT=7a089fd0d7dafa00cc6116ea714bf6fed9d0743e
RUN git clone https://github.com/ostris/ai-toolkit.git /ai-toolkit && \
    cd /ai-toolkit && \
    git checkout ${AI_TOOLKIT_COMMIT} && \
    git submodule update --init --recursive

# Install AI-Toolkit requirements — these own the transformers/diffusers/peft pins
RUN pip install --no-cache-dir --break-system-packages -r /ai-toolkit/requirements.txt

# Re-pin the matched cu130 torch trio AFTER the AI-Toolkit requirements install —
# torchao/torchcodec/etc. can re-resolve torch or torchaudio off the default PyPI
# index and break the ABI match. --no-deps keeps the already-installed nvidia libs.
RUN pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
    torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

# ============================================================
# Flash Attention 3 (Hopper-only — H100/H200, CUDA 13.0)
# ============================================================
RUN pip install --no-cache-dir --break-system-packages \
    flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch291

# Flash Attention 2 (non-Hopper fallback) — PINNED prebuilt wheel, NEVER compiled.
# Installed by direct .whl URL so pip installs it as-is: no PyPI resolution (which
# would prefer a newer sdist and compile from source → OOMs CI), no source build.
# Matches the image exactly: flash_attn 2.8.3 / cu130 / torch2.9 / cp312 / linux_x86_64.
RUN pip install --no-cache-dir --break-system-packages \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.9-cp312-cp312-linux_x86_64.whl

# ============================================================
# HuggingFace CLI (standalone binary for model downloads)
# ============================================================
RUN curl -LsSf https://hf.co/cli/install.sh | bash

# ============================================================
# Verify critical imports at build time
# ============================================================
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" && \
    python -c "import torchaudio; print(f'torchaudio {torchaudio.__version__} OK')" && \
    python -c "import torchvision; print(f'torchvision {torchvision.__version__} OK')" && \
    python -c "import transformers; print(f'Transformers {transformers.__version__}')" && \
    python -c "import runpod; print('RunPod OK')" && \
    python -c "import boto3; print('boto3 OK')" && \
    python -c "exec('try:\n from flash_attn_3 import flash_attn_func; print(\"Flash Attention 3 OK\")\nexcept:\n print(\"Flash Attention 3 not available\")')" && \
    python -c "exec('try:\n from flash_attn import flash_attn_func; print(\"Flash Attention 2 OK\")\nexcept:\n print(\"Flash Attention 2 not available\")')" && \
    cd /ai-toolkit && python -c "from toolkit.job import get_job; print('AI-Toolkit OK')"

# ============================================================
# Entrypoint — bake the worker code into the image (no runtime clone)
# ============================================================
# COPY is the LAST layer so editing worker code doesn't bust the heavy pip
# cache above. .dockerignore keeps tests/docs/CI out of the image.
COPY . /app/runtime
# Explicit handler copy so the RunPod Hub handler-detector sees handler.py
# (already included by `COPY .` above; this line is for static detection).
COPY handler.py /app/runtime/handler.py
WORKDIR /app/runtime

# Worker boots via start.sh (env + torch/aitk pin), which runs `python handler.py`.
CMD ["bash", "start.sh"]
