"""Translate diffusion-pipe dot-path config_overrides → a partial AI-Toolkit
config dict (nested under config.process[0]).

Closed allow-list. Unknown keys, and known-but-deliberately-dropped keys, are
LOGGED (alog.warn) and IGNORED — this function NEVER raises (DESIGN §I-D2). The
only hard-fail validation lives in config.validate_payload.

Field names / sinks are byte-exact from SOURCE_FINDINGS.md §5 (cited to
/tmp/aitk_src @ 7a089fd). Notable source corrections vs the original draft:
  - gradient_clipping  → train.max_grad_norm        (NOT optimizer_params.clip_grad_norm — that field doesn't exist)
  - warmup_steps       → train.lr_scheduler_params.num_warmup_steps + lr_scheduler='constant_with_warmup'
  - optimizer.eps      → MANDATORY DROP (forwarding it CRASHES: eps is hardcoded in the
                         optimizer ctor and optimizer_params is splatted → TypeError)
  - gradient_accumulation_steps → train.gradient_accumulation (the proper-accumulation key)
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger as alog


# ---------------------------------------------------------------------------
# Allow-list  (dot-path → nested sink path under config.process[0])
# Each entry: (sink_path_tuple, transform_callable_or_None).
# A transform maps the raw value → the value to set at the sink (or returns a
# dict fragment to merge when one source key fans out to several sinks).
# ---------------------------------------------------------------------------

def _passthrough(v: Any) -> Any:
    return v


def _optimizer_type(v: Any) -> str:
    # adamw_optimi is not a valid AI-Toolkit optimizer; map to adamw8bit
    # (numerical delta accepted, R8). Passthrough adamw8bit/adamw.
    if v == "adamw_optimi":
        return "adamw8bit"
    return v


# Simple 1:1 renames: source dot-path -> sink path tuple (rooted at process[0]).
_RENAME_SINKS: dict[str, tuple[str, ...]] = {
    "optimizer.lr": ("train", "lr"),
    "optimizer.betas": ("train", "optimizer_params", "betas"),
    "optimizer.weight_decay": ("train", "optimizer_params", "weight_decay"),
    "gradient_accumulation_steps": ("train", "gradient_accumulation"),
    "activation_checkpointing": ("train", "gradient_checkpointing"),
    "micro_batch_size_per_gpu": ("train", "batch_size"),
    "save_dtype": ("save", "dtype"),
    "gradient_clipping": ("train", "max_grad_norm"),
    "model.unet_lr": ("train", "unet_lr"),
}

# Renames needing a value transform.
_TRANSFORM_SINKS: dict[str, tuple[tuple[str, ...], Any]] = {
    "optimizer.type": (("train", "optimizer"), _optimizer_type),
}

# Keys consumed by the epoch→step math (handled specially, not direct sinks).
_EPOCH_KEYS = {"epochs", "save_every_n_epochs"}
# num_repeats both feeds epoch math AND sets a dataset field.
_DATASET_NUM_REPEATS = "dataset.num_repeats"

# Known keys we deliberately DROP (warn + ignore). Rationale per row in DESIGN §C.
_DROPPED_KEYS = {
    "adapter.dtype",              # re-expressed as model.quantize
    "optimizer.eps",             # MANDATORY drop — forwarding crashes (§5)
    "compile",                   # no clean AI-Toolkit bool; wan already false
    "pipeline_stages",           # single-GPU; no pipeline parallel
    "model.diffusion_model", "model.vae", "model.text_encoders",
    "model.transformer_path", "model.ckpt_path", "model.diffusers_path",
    "model.checkpoint_path",     # infra paths owned by ModelSpec; client path ignored
    "model.transformer_dtype", "model.diffusion_model_dtype",  # quant owned by ModelSpec
    "model.min_t", "model.max_t",            # wan window — handled by noise_variant
    "model.merge_adapters",      # z_image turbo owned by ModelSpec
    "training_strategy.with_audio",          # LTX out of scope
    "dataset.enable_ar_bucket", "dataset.min_ar", "dataset.max_ar",
    "dataset.num_ar_buckets",    # AI-Toolkit auto-buckets by resolution list
}


def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set d[path[0]][path[1]]... = value, creating intermediate dicts."""
    target = d
    for part in path[:-1]:
        target = target.setdefault(part, {})
    target[path[-1]] = value


def _set_dataset0(out: dict[str, Any], field: str, value: Any) -> None:
    """Set datasets[0][field]=value. `datasets` is a single-element LIST so it
    deep-merges element-wise with the template's parsed `datasets: [ {...} ]`."""
    datasets = out.setdefault("datasets", [{}])
    datasets[0][field] = value


# ---------------------------------------------------------------------------
# Epoch ↔ step math (DESIGN §B3) — reproduces diffusion-pipe _calculate_total_steps
# minus the multi-GPU data_parallel_degree term (always 1 now).
# ---------------------------------------------------------------------------

def steps_per_epoch(dataset_image_count: int, num_repeats: int,
                    effective_batch: int) -> int:
    """ceil(images * num_repeats / effective_batch). At least 1."""
    return max(1, math.ceil(dataset_image_count * num_repeats / max(1, effective_batch)))


def translate_overrides(
    overrides: dict[str, Any],
    *,
    model_type: str,
    dataset_image_count: int,
    base_batch_size: int,
    base_grad_accum: int,
) -> tuple[dict[str, Any], list[str]]:
    """Map diffusion-pipe dot-path overrides → a partial nested AI-Toolkit config
    dict (rooted at the process[0] level: keys are 'train','save','network',
    'datasets','model').

    Returns (partial_config, ignored_keys) where ignored_keys is the list of
    keys that were warned + dropped (known-but-unsupported) or warned + ignored
    (genuinely unknown). The caller LOGS ignored_keys (alog.warn) but does NOT
    surface them in the OUTPUT JSON — the response shape is frozen (N3, lead RULED).

    NEVER raises. Reads no filesystem.
    """
    out: dict[str, Any] = {}
    ignored: list[str] = []

    # ---- resolve the effective batch (base ⊕ override) for epoch math ----
    batch_size = base_batch_size
    grad_accum = base_grad_accum
    if "micro_batch_size_per_gpu" in overrides:
        batch_size = _coerce_int(overrides["micro_batch_size_per_gpu"], batch_size)
    if "gradient_accumulation_steps" in overrides:
        grad_accum = _coerce_int(overrides["gradient_accumulation_steps"], grad_accum)
    effective_batch = max(1, batch_size * grad_accum)

    num_repeats = 1
    if _DATASET_NUM_REPEATS in overrides:
        num_repeats = _coerce_int(overrides[_DATASET_NUM_REPEATS], 1)

    spe = steps_per_epoch(dataset_image_count, num_repeats, effective_batch)

    # ---- walk every override key through the closed allow-list ----
    for key, value in overrides.items():
        if key in _DROPPED_KEYS:
            alog.warning(f"override key '{key}' is not supported on AI-Toolkit; "
                         "ignoring (did not take effect)")
            ignored.append(key)
            continue

        if key in _EPOCH_KEYS:
            if key == "epochs":
                _set_nested(out, ("train", "steps"), _coerce_int(value, 1) * spe)
            else:  # save_every_n_epochs
                _set_nested(out, ("save", "save_every"), _coerce_int(value, 1) * spe)
            continue

        if key == _DATASET_NUM_REPEATS:
            _set_dataset0(out, "num_repeats", _coerce_int(value, 1))
            continue

        if key == "dataset.resolutions":
            # plural→singular; list passthrough (auto-bucketed by AI-Toolkit)
            _set_dataset0(out, "resolution", value)
            continue

        if key == "dataset.frame_buckets":
            # map first/only frame bucket → num_frames (image=1)
            frames = value[0] if isinstance(value, list) and value else value
            _set_dataset0(out, "num_frames", _coerce_int(frames, 1))
            continue

        if key == "adapter.rank":
            rank = _coerce_int(value, 32)
            _set_nested(out, ("network", "linear"), rank)
            _set_nested(out, ("network", "linear_alpha"), rank)  # alpha == rank
            continue

        if key == "adapter.type":
            if value != "lora":
                alog.warning(f"adapter.type '{value}' unsupported; keeping 'lora'")
            _set_nested(out, ("network", "type"), "lora")
            continue

        if key == "warmup_steps":
            # actual sink is lr_scheduler_params.num_warmup_steps and only honored
            # with lr_scheduler='constant_with_warmup' (SOURCE_FINDINGS §5).
            _set_nested(out, ("train", "lr_scheduler"), "constant_with_warmup")
            _set_nested(out, ("train", "lr_scheduler_params", "num_warmup_steps"),
                        _coerce_int(value, 0))
            continue

        if key in ("model.text_encoder_1_lr", "model.text_encoder_2_lr"):
            # both collapse to a single train.text_encoder_lr (minor lossiness, R-SDXL).
            # prefer te_1; if both present and differ, warn.
            te1 = overrides.get("model.text_encoder_1_lr")
            te2 = overrides.get("model.text_encoder_2_lr")
            chosen = te1 if te1 is not None else te2
            if te1 is not None and te2 is not None and te1 != te2:
                alog.warning("model.text_encoder_{1,2}_lr differ; collapsing to one "
                             "train.text_encoder_lr (using te_1)")
            _set_nested(out, ("train", "text_encoder_lr"), chosen)
            continue

        if key in _RENAME_SINKS:
            _set_nested(out, _RENAME_SINKS[key], value)
            continue

        if key in _TRANSFORM_SINKS:
            sink, fn = _TRANSFORM_SINKS[key]
            _set_nested(out, sink, fn(value))
            continue

        # genuinely unknown — warn + ignore (never raise).
        alog.warning(f"unknown override key '{key}' did not take effect; ignoring")
        ignored.append(key)

    return out, ignored


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
