"""Build the single AI-Toolkit config.yml for a training job.

Mirrors the old toml_generator flow (load template → substitute SYSTEM paths →
parse → apply translated overrides → dump), with TOML→YAML the only format
swap. The AI-Toolkit config is nested under `config.process[0]` with a top-level
`job: extension` (SOURCE_FINDINGS §0); the deep-merge root is
cfg["config"]["process"][0].

N1 (DESIGN §I-N1): `{placeholder}` substitution is for SYSTEM PATHS ONLY. All
user-controlled values (trigger_word, every config_override) are assigned onto
the PARSED dict AFTER yaml.safe_load — they never pass through string
substitution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger as alog

import config as cfg
from config import (
    AITK_JOB_NAME,
    MAX_SAVES_KEEP,
    ModelSpec,
    TrainingJob,
)
from overrides_translator import steps_per_epoch, translate_overrides

CONFIGS_PACKAGE_DIR = Path(__file__).parent / "configs"

# model_type → template filename (qwen_image_2512 reuses qwen_image arch but has
# its own template for the repo/qtype swap).
_TEMPLATE_MAP = {
    "wan2.2": "wan2.2.yaml",
    "sdxl": "sdxl.yaml",
    "qwen_image": "qwen_image.yaml",
    "qwen_image_2512": "qwen_image_2512.yaml",
    "z_image": "z_image.yaml",
    "ideogram4": "ideogram4.yaml",
    "flux_klein_9b": "flux_klein_9b.yaml",
    "krea2": "krea2.yaml",
    "krea2_turbo": "krea2_turbo.yaml",
}


def _load_template(template_name: str) -> str:
    with open(CONFIGS_PACKAGE_DIR / template_name) as f:
        return f.read()


def _substitute_paths(raw: str, substitutions: dict[str, str]) -> str:
    """Replace {placeholders} in the raw template — SYSTEM PATHS ONLY (N1)."""
    result = raw
    for key, value in substitutions.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _resolve_name_or_path(job: TrainingJob, spec: ModelSpec) -> str:
    """Resolve the SYSTEM-owned model.name_or_path for Phase-1 substitution.

    This NEVER returns a user-controlled value — the CivitAI checkpoint path
    (which can contain ':'/spaces) is assigned POST-parse in Phase 2 (N1
    hardening), not substituted into raw template text.

    sdxl: the local base single-file (a file path triggers from_single_file —
    SOURCE_FINDINGS §8). Everything else: the local download dir under MODELS_DIR
    (falls back to the repo id if the dir is absent — AI-Toolkit fetches from HF).
    """
    if job.model_type == "sdxl":
        return str(cfg.MODELS_DIR / "sdxl-base-1.0" / "sd_xl_base_1.0_0.9vae.safetensors")

    # local dir for the (single) repo download; else the repo id.
    for item in spec.downloads:
        if item.kind == "repo" and item.local_subdir:
            local = cfg.MODELS_DIR / item.local_subdir
            return str(local) if local.exists() else (item.repo_id or spec.name_or_path)
    return spec.name_or_path


def _deep_merge(base: Any, patch: Any) -> Any:
    """Recursively merge patch into base. Dicts merge key-wise; lists merge
    element-wise by index (so datasets[0] partials land on the template's
    datasets[0]); scalars and mismatched types are overwritten by patch."""
    if isinstance(base, dict) and isinstance(patch, dict):
        for k, v in patch.items():
            base[k] = _deep_merge(base.get(k), v) if k in base else v
        return base
    if isinstance(base, list) and isinstance(patch, list):
        merged = list(base)
        for i, v in enumerate(patch):
            if i < len(merged):
                merged[i] = _deep_merge(merged[i], v)
            else:
                merged.append(v)
        return merged
    return patch


def generate_config(job: TrainingJob, dataset_image_count: int) -> Path:
    """Build the AI-Toolkit config.yml for this job; return its path.

    Steps: load template → substitute SYSTEM paths → yaml.safe_load → assign
    user values (trigger_word) + translated overrides onto the parsed dict
    (deep-merged at process[0]) → compute steps/save_every from epoch semantics
    → write config.yml.
    """
    job.configs_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    spec = job.model_spec

    raw = _load_template(_TEMPLATE_MAP[job.model_type])

    # ---- phase 1: SYSTEM-PATH substitution only (N1) ----
    name_or_path = _resolve_name_or_path(job, spec)
    raw = _substitute_paths(raw, {
        "name_or_path": name_or_path,
        "training_folder": str(job.output_dir),
        "dataset_path": str(job.dataset_dir),
    })

    # ---- phase 2: parse, then assign user-controlled values onto the dict ----
    full = yaml.safe_load(raw)
    proc = full["config"]["process"][0]  # deep-merge root (SOURCE_FINDINGS §0)

    proc["name"] = AITK_JOB_NAME  # fixed; trigger never used as the config name
    full["config"]["name"] = AITK_JOB_NAME

    # trigger_word is a user value → assigned post-parse (N1), native AI-Toolkit
    # caption prepend (no .txt edits) — SOURCE_FINDINGS §7.
    proc["datasets"][0]["trigger_word"] = job.trigger_word
    proc["datasets"][0]["caption_ext"] = "txt"

    # N1 hardening: the CivitAI checkpoint path is USER-controlled (may contain
    # ':'/spaces) → assigned onto the parsed dict here, NEVER substituted into raw
    # template text. A file path triggers AI-Toolkit's from_single_file (§8).
    if job.model_type == "sdxl" and job.civitai_checkpoint_path:
        proc.setdefault("model", {})["name_or_path"] = job.civitai_checkpoint_path

    # keep ALL checkpoints: a large positive int (NOT -1, which deletes all but
    # the oldest — SOURCE_FINDINGS §6).
    proc.setdefault("save", {})["max_step_saves_to_keep"] = MAX_SAVES_KEEP

    # wan2.2 single-flag: exactly ONE expert flag true; split_multistage_loras
    # stays true → one suffixed file (SOURCE_FINDINGS §3).
    if job.is_wan22:
        model = proc.setdefault("model", {})
        mk = model.setdefault("model_kwargs", {})
        mk["train_high_noise"] = job.noise_variant == "high"
        mk["train_low_noise"] = job.noise_variant == "low"
        proc.setdefault("network", {})["split_multistage_loras"] = True

    # ---- epoch→step base math + translated overrides ----
    base_batch = proc.get("train", {}).get("batch_size", 1)
    base_accum = proc.get("train", {}).get("gradient_accumulation", 1)

    patch, ignored_keys = translate_overrides(
        job.config_overrides,
        model_type=job.model_type,
        dataset_image_count=dataset_image_count,
        base_batch_size=base_batch,
        base_grad_accum=base_accum,
    )
    # N3 (lead RULED): ignored override keys are LOGGED only — never added to the
    # OUTPUT JSON (response shape frozen). translate_overrides already alog.warn'd
    # each; emit a single rollup for visibility.
    if ignored_keys:
        alog.warning(f"ignored {len(ignored_keys)} unsupported override key(s): "
                     f"{ignored_keys}")
    _deep_merge(proc, patch)

    # If epochs/save_every were NOT overridden, compute train.steps and
    # save.save_every from the model defaults so they always have concrete
    # step values (DESIGN §B3). num_repeats from the (post-merge) dataset.
    num_repeats = proc["datasets"][0].get("num_repeats", 1)
    eff_batch = max(1, proc["train"].get("batch_size", base_batch)
                    * proc["train"].get("gradient_accumulation", base_accum))
    spe = steps_per_epoch(dataset_image_count, num_repeats, eff_batch)
    if "steps" not in patch.get("train", {}):
        proc["train"]["steps"] = spec.defaults.epochs * spe
    if "save_every" not in patch.get("save", {}):
        proc["save"]["save_every"] = spec.defaults.save_every_n_epochs * spe

    proc["training_folder"] = str(job.output_dir)
    proc["device"] = "cuda:0"

    out_path = job.configs_dir / "config.yml"
    with open(out_path, "w") as f:
        yaml.safe_dump(full, f, sort_keys=False)

    alog.info(f"Generated AI-Toolkit config: {out_path}")
    return out_path
