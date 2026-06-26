"""GPU detection and adaptive configuration."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from loguru import logger


@dataclass
class GPUInfo:
    count: int
    name: str
    vram_gb: float

    @property
    def total_vram_gb(self) -> float:
        return self.count * self.vram_gb

    @property
    def is_high_vram(self) -> bool:
        """Per-GPU VRAM >= 100 GB (H200, B200)."""
        return self.vram_gb >= 100

    @property
    def is_hopper_or_newer(self) -> bool:
        """H100, H200, B200, or newer."""
        hopper_plus = {"H100", "H200", "B200", "B100", "GB200", "GB300"}
        return any(gpu in self.name for gpu in hopper_plus)


def detect_gpus() -> GPUInfo:
    """Detect GPU count, name, and VRAM from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi failed, assuming 1 GPU")
            return GPUInfo(count=1, name="unknown", vram_gb=80)

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        count = len(lines)
        # Parse first GPU (assume homogeneous)
        parts = lines[0].split(",")
        name = parts[0].strip()
        vram_mb = float(parts[1].strip())
        vram_gb = round(vram_mb / 1024, 1)

        info = GPUInfo(count=count, name=name, vram_gb=vram_gb)
        logger.info(f"Detected {count}x {name} ({vram_gb} GB each, {info.total_vram_gb} GB total)")
        return info

    except Exception as e:
        logger.warning(f"GPU detection failed: {e}, assuming 1 GPU")
        return GPUInfo(count=1, name="unknown", vram_gb=80)


# Singleton — detected once at import time
_gpu_info: GPUInfo | None = None


def get_gpu_info() -> GPUInfo:
    global _gpu_info
    if _gpu_info is None:
        _gpu_info = detect_gpus()
    return _gpu_info
