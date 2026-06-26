"""Parse training stderr/stdout for progress metrics and report to RunPod.

AI-Toolkit emits tqdm bars on STDERR (no \n until close — trainer reader must
split on both \r and \n). The canonical line format is:

    aitk_lora:  12%|█▎        | 250/2000 [01:23<09:45,  2.99it/s, lr: 1.0e-04 loss: 1.234e-01]

STEP_LOSS_PATTERN matches the N/total pair and the loss postfix in one pass.
STEP_PATTERN is a fallback for lines that have N/total but no loss yet (e.g.
the very first bar render before the postfix is set).
LOSS_PATTERN is a standalone fallback for any other line that carries only loss.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

from loguru import logger

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Primary: AI-Toolkit tqdm bar — matches N/total + loss postfix in one pass.
# Anchors: `\d+/\d+` avoids grabbing lr (lr has no slash); `loss[:=]\s*` with
# optional space handles the `loss: 1.234e-01` postfix format exactly.
# SOURCE_FINDINGS §2: postfix is `lr: 1.0e-04 loss: 1.234e-01`; loss is %.3e.
STEP_LOSS_PATTERN = re.compile(
    r"(\d+)\s*/\s*(\d+).*?loss\s*[:=]\s*([\d.eE+-]+)"
)

# Fallback: N/total without loss (first bar render or lines missing postfix).
STEP_PATTERN = re.compile(r"(\d+)\s*/\s*(\d+)")

# Standalone loss (any line carrying loss= or loss: outside a tqdm bar).
LOSS_PATTERN = re.compile(r"(?:avr_loss|loss)\s*[=:]\s*([\d.eE+-]+)")

THROTTLE_INTERVAL = 10  # seconds between RunPod progress updates


# ---------------------------------------------------------------------------
# Progress dataclass — public contract (frozen interface, spec §B)
# ---------------------------------------------------------------------------

@dataclass
class TrainingProgress:
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    total_epochs: int = 0
    steps_per_epoch: int = 0   # pre-set by the trainer; used to synthesize epoch
    loss: float | None = None
    stage: str = "initializing"
    percent: float = 0.0
    job_id: str | None = None
    label: str = ""
    ready_loras: list[dict] = field(default_factory=list)
    _last_report_time: float = field(default_factory=time.time)

    @property
    def progress_dict(self) -> dict:
        d = {
            "step": self.step,
            "total_steps": self.total_steps,
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "stage": self.stage,
            "percent": round(self.percent, 1),
        }
        if self.loss is not None:
            d["loss"] = round(self.loss, 6)
        if self.label:
            d["label"] = self.label
        if self.ready_loras:
            d["ready_loras"] = self.ready_loras
        return d


# ---------------------------------------------------------------------------
# parse_line — called per \r- or \n-delimited fragment from the trainer reader
# ---------------------------------------------------------------------------

def _sync_epoch(p: TrainingProgress) -> None:
    """Synthesize the current epoch from the step — AI-Toolkit is step-based and
    emits no epoch line. ceil() = 'currently in epoch N', capped at total_epochs."""
    if p.steps_per_epoch > 0 and p.step > 0:
        ep = math.ceil(p.step / p.steps_per_epoch)
        p.epoch = min(ep, p.total_epochs) if p.total_epochs else ep


def parse_line(line: str, progress: TrainingProgress) -> bool:
    """Parse a single stderr/stdout fragment for training metrics.

    The caller (trainer._drain_stderr) is responsible for splitting the raw
    stream on both \\r and \\n before passing fragments here (F5).

    Returns True if any field on progress was updated.
    """
    # Primary path: tqdm bar with step N/total AND loss in one line.
    match = STEP_LOSS_PATTERN.search(line)
    if match:
        progress.step = int(match.group(1))
        bar_total = int(match.group(2))
        # Prefer the caller-supplied total_steps (pre-calculated from job
        # config); adopt the bar's denominator only when total is unknown.
        if progress.total_steps == 0 and bar_total > 0:
            progress.total_steps = bar_total
        progress.stage = "training"
        if progress.total_steps > 0:
            progress.percent = progress.step / progress.total_steps * 100
        try:
            progress.loss = float(match.group(3))
        except ValueError:
            pass
        _sync_epoch(progress)
        return True

    # Fallback: bar has N/total but postfix not yet populated.
    match = STEP_PATTERN.search(line)
    if match:
        progress.step = int(match.group(1))
        bar_total = int(match.group(2))
        if progress.total_steps == 0 and bar_total > 0:
            progress.total_steps = bar_total
        progress.stage = "training"
        if progress.total_steps > 0:
            progress.percent = progress.step / progress.total_steps * 100
        _sync_epoch(progress)
        return True

    # Standalone loss line (no N/total context).
    match = LOSS_PATTERN.search(line)
    if match:
        try:
            progress.loss = float(match.group(1))
            return True
        except ValueError:
            pass

    return False


# ---------------------------------------------------------------------------
# Throttled reporting helpers
# ---------------------------------------------------------------------------

def should_report(progress: TrainingProgress) -> bool:
    """Return True if enough time has elapsed to send a progress report."""
    now = time.time()
    if now - progress._last_report_time >= THROTTLE_INTERVAL:
        progress._last_report_time = now
        return True
    return False


def report_progress(progress: TrainingProgress, label: str = "") -> None:
    """Log progress and send to RunPod if job_id is set."""
    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}Progress: stage={progress.stage} "
        f"step={progress.step}/{progress.total_steps} "
        f"epoch={progress.epoch} loss={progress.loss} "
        f"percent={progress.percent:.1f}%"
    )

    if progress.job_id:
        try:
            import runpod

            runpod.serverless.progress_update(
                {"id": progress.job_id},
                progress.progress_dict,
            )
        except Exception:
            pass  # Don't crash training over a progress update failure
