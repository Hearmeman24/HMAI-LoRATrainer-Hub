#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# HMAI LoRA Trainer — Runtime startup
# Image entrypoint (CMD). Worker code is baked at /app/runtime.
# ============================================================

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "${VOLUME_ROOT}/jobs" "${VOLUME_ROOT}/models" "${VOLUME_ROOT}/hf_cache" "${VOLUME_ROOT}/logs"

DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs 2>/dev/null || echo "unknown")

# ============================================================
# Environment setup
# ============================================================
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${VOLUME_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
# hf_xet acceleration for huggingface_hub downloads (model_downloader.py)
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH="${RUNTIME_DIR}:${PYTHONPATH:-}"
# Full-dataset image runs (800+ pairs, 60+ epochs) exceed the 4h code
# default; endpoint REST API rejects env updates so the floor lives here.
# executionTimeoutMs on the endpoint (16.7h) remains the hard backstop.
export MAX_TRAINING_HOURS="${MAX_TRAINING_HOURS:-12}"

# ============================================================
# Pin /ai-toolkit to a runtime SHA (allows hotfix without rebuild)
# ============================================================
# The image bakes AI-Toolkit at AI_TOOLKIT_COMMIT (Dockerfile ARG). This
# block fast-forwards the on-disk clone when a newer ref is needed without
# a full image rebuild. Default matches the baked SHA so cold starts are
# a no-op. Requirements.txt is NOT re-run here — only code changes, not
# dep changes, are expected from a hotfix SHA bump.
#
# Never abort the container on failure: on any error, fall back to the
# baked tree. Jobs will still run; only the new ref's fixes are missing.
AI_TOOLKIT_REF="${AI_TOOLKIT_REF:-7a089fd0d7dafa00cc6116ea714bf6fed9d0743e}"
set +e
if [ -n "$AI_TOOLKIT_REF" ] && [ -d /ai-toolkit/.git ]; then
    CURRENT_AITK_REF=$(git -C /ai-toolkit rev-parse HEAD 2>/dev/null)
    if [ "$CURRENT_AITK_REF" != "$AI_TOOLKIT_REF" ]; then
        echo "[aitk-pin] updating /ai-toolkit ${CURRENT_AITK_REF:0:12} -> ${AI_TOOLKIT_REF:0:12}"
        if git -C /ai-toolkit fetch --depth 1 origin "$AI_TOOLKIT_REF" \
            && git -C /ai-toolkit checkout --force FETCH_HEAD; then
            echo "[aitk-pin] /ai-toolkit now at $(git -C /ai-toolkit rev-parse HEAD)"
        else
            echo "[aitk-pin] WARNING: update failed — rolling back to baked tree ${CURRENT_AITK_REF:0:12}"
            git -C /ai-toolkit checkout --force "$CURRENT_AITK_REF" \
                || echo "[aitk-pin] WARNING: rollback also failed — tree may be inconsistent"
        fi
    else
        echo "[aitk-pin] /ai-toolkit already at ${AI_TOOLKIT_REF:0:12}"
    fi
else
    echo "[aitk-pin] skipped (no ref or /ai-toolkit is not a git repo)"
fi
set -e

# ============================================================
# Matched torch trio (cu130) — torchaudio ABI hotfix
# ============================================================
# AI-Toolkit imports torchaudio at module load (toolkit/config_modules.py). The
# image pins torch==2.9.1 but left torchvision/torchaudio unpinned, and the
# AI-Toolkit requirements install can leave torchaudio built against a different
# torch -> `undefined symbol: torch_library_impl` at import, which kills every
# training run. Force the matched cu130 trio here (runs AFTER all image pip
# layers, so it corrects any drift). Self-deactivating: if torchaudio already
# imports, this no-ops — so once the fix is baked into the image it costs nothing.
# RUNTIME HOTFIX to validate the fix on the live v3.0.1 image; to be baked into
# the Dockerfile and removed from here.
set +e
if python3 -c "import torchaudio" 2>/dev/null; then
    echo "[torch-pin] torchaudio imports OK — no reinstall needed"
else
    echo "[torch-pin] torchaudio import broken — reinstalling matched cu130 trio (torch 2.9.1 / torchvision 0.24.1 / torchaudio 2.9.1)..."
    pip install --no-cache-dir --break-system-packages --force-reinstall --no-deps \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu130 \
        && python3 -c "import torchaudio; print('[torch-pin] torchaudio import OK after reinstall')" \
        || echo "[torch-pin] WARNING: torch trio reinstall failed — training will likely hit the torchaudio ABI error"
fi
set -e

echo ""
echo "================================================"
echo "  HMAI LoRA Trainer ready!"
echo "  Runtime: ${RUNTIME_DIR}"
echo "  Volume:  ${VOLUME_ROOT}"
echo "  GPU:     ${DETECTED_GPU}"
echo "================================================"
echo ""

# ============================================================
# Launch handler
# ============================================================
cd "$RUNTIME_DIR"
exec python handler.py
