"""Tests for yaml_generator.py — per-model base shape, deep-merge, N1 YAML-hostile
trigger words, wan2.2 single-flag, step/save_every computation, config.name invariants.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import config as cfg
from config import (
    AITK_JOB_NAME,
    MAX_SAVES_KEEP,
    TrainingJob,
    validate_payload,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_jobs_dir(tmp_path):
    original = cfg.JOBS_DIR
    cfg.JOBS_DIR = tmp_path / "jobs"
    yield
    cfg.JOBS_DIR = original


@pytest.fixture(autouse=True)
def _patch_models_dir(tmp_path):
    original = cfg.MODELS_DIR
    cfg.MODELS_DIR = tmp_path / "models"
    yield
    cfg.MODELS_DIR = original


def _make_job(model_type: str, noise_variant: str | None = None,
              trigger_word: str = "testword", **extra) -> TrainingJob:
    payload: dict = {
        "model_type": model_type,
        "dataset_zip_url": "https://example.com/data.zip",
        "trigger_word": trigger_word,
        "job_id": "test-job-001",
    }
    if noise_variant is not None:
        payload["noise_variant"] = noise_variant
    payload.update(extra)
    return validate_payload(payload)


def _generate(job: TrainingJob, img_count: int = 10) -> dict:
    """Run generate_config and return the parsed YAML dict."""
    from yaml_generator import generate_config
    out_path = generate_config(job, img_count)
    with open(out_path) as f:
        return yaml.safe_load(f)


def _proc(full: dict) -> dict:
    """Return the process[0] dict — the deep-merge root."""
    return full["config"]["process"][0]


# ---------------------------------------------------------------------------
# Top-level structure invariants
# ---------------------------------------------------------------------------

class TestTopLevelShape:
    def test_job_extension(self):
        job = _make_job("sdxl")
        full = _generate(job)
        assert full["job"] == "extension"

    def test_config_name_is_aitk_lora(self):
        """config.name must be AITK_JOB_NAME, never the trigger word."""
        job = _make_job("sdxl", trigger_word="my_custom_trigger")
        full = _generate(job)
        assert full["config"]["name"] == AITK_JOB_NAME
        assert full["config"]["name"] == "aitk_lora"

    def test_process_is_list(self):
        job = _make_job("sdxl")
        full = _generate(job)
        assert isinstance(full["config"]["process"], list)
        assert len(full["config"]["process"]) >= 1

    def test_proc_name_is_aitk_lora(self):
        """process[0].name == 'aitk_lora' (fixed job name, F3)."""
        job = _make_job("sdxl")
        proc = _proc(_generate(job))
        assert proc["name"] == AITK_JOB_NAME

    def test_max_step_saves_to_keep_is_large_not_minus_one(self):
        """max_step_saves_to_keep must be MAX_SAVES_KEEP=100000, never -1 (F2)."""
        job = _make_job("sdxl")
        proc = _proc(_generate(job))
        assert proc["save"]["max_step_saves_to_keep"] == MAX_SAVES_KEEP
        assert proc["save"]["max_step_saves_to_keep"] == 100000
        assert proc["save"]["max_step_saves_to_keep"] != -1

    def test_device_is_cuda0(self):
        job = _make_job("sdxl")
        proc = _proc(_generate(job))
        assert proc["device"] == "cuda:0"

    def test_trigger_word_assigned_to_dataset(self):
        job = _make_job("sdxl", trigger_word="testword")
        proc = _proc(_generate(job))
        assert proc["datasets"][0]["trigger_word"] == "testword"

    def test_caption_ext_is_txt(self):
        job = _make_job("sdxl")
        proc = _proc(_generate(job))
        assert proc["datasets"][0]["caption_ext"] == "txt"


# ---------------------------------------------------------------------------
# Per-model base shape
# ---------------------------------------------------------------------------

class TestPerModelBaseShape:
    def test_sdxl_arch(self):
        proc = _proc(_generate(_make_job("sdxl")))
        assert proc["model"]["arch"] == "sdxl"

    def test_sdxl_no_quantize(self):
        proc = _proc(_generate(_make_job("sdxl")))
        assert proc["model"].get("quantize") is False or not proc["model"].get("quantize", True)

    def test_qwen_image_arch(self):
        proc = _proc(_generate(_make_job("qwen_image")))
        assert proc["model"]["arch"] == "qwen_image"

    def test_qwen_image_resolution(self):
        """qwen_image forced resolution [1328]."""
        proc = _proc(_generate(_make_job("qwen_image")))
        assert proc["datasets"][0]["resolution"] == [1328]

    def test_qwen_image_2512_arch(self):
        proc = _proc(_generate(_make_job("qwen_image_2512")))
        assert proc["model"]["arch"] == "qwen_image"

    def test_z_image_arch_no_underscore(self):
        """z_image arch must be 'zimage' — NO underscore (SOURCE_FINDINGS §4)."""
        proc = _proc(_generate(_make_job("z_image")))
        assert proc["model"]["arch"] == "zimage"
        assert proc["model"]["arch"] != "z_image"

    def test_ideogram4_arch(self):
        proc = _proc(_generate(_make_job("ideogram4")))
        assert proc["model"]["arch"] == "ideogram4"

    def test_ideogram4_timestep_type_linear(self):
        proc = _proc(_generate(_make_job("ideogram4")))
        assert proc["train"].get("timestep_type") == "linear"

    def test_flux_klein_9b_arch(self):
        """flux_klein_9b arch must be 'flux2_klein_9b' (NOT 'flux2_klein')."""
        proc = _proc(_generate(_make_job("flux_klein_9b")))
        assert proc["model"]["arch"] == "flux2_klein_9b"
        assert proc["model"]["arch"] != "flux2_klein"

    def test_flux_klein_9b_weighted_timestep(self):
        proc = _proc(_generate(_make_job("flux_klein_9b")))
        assert proc["train"].get("timestep_type") == "weighted"


# ---------------------------------------------------------------------------
# wan2.2 single-flag mechanism
# ---------------------------------------------------------------------------

class TestWan22SingleFlag:
    def test_high_noise_sets_train_high_only(self):
        proc = _proc(_generate(_make_job("wan2.2", noise_variant="high")))
        mk = proc["model"]["model_kwargs"]
        assert mk["train_high_noise"] is True
        assert mk["train_low_noise"] is False

    def test_low_noise_sets_train_low_only(self):
        proc = _proc(_generate(_make_job("wan2.2", noise_variant="low")))
        mk = proc["model"]["model_kwargs"]
        assert mk["train_high_noise"] is False
        assert mk["train_low_noise"] is True

    def test_split_multistage_loras_true(self):
        proc = _proc(_generate(_make_job("wan2.2", noise_variant="high")))
        assert proc["network"].get("split_multistage_loras") is True

    def test_wan22_is_one_model_kwargs_call(self):
        """Only one of high/low is true — never both."""
        for variant in ("high", "low"):
            proc = _proc(_generate(_make_job("wan2.2", noise_variant=variant)))
            mk = proc["model"]["model_kwargs"]
            assert sum([mk["train_high_noise"], mk["train_low_noise"]]) == 1


# ---------------------------------------------------------------------------
# Steps / save_every computation
# ---------------------------------------------------------------------------

class TestStepComputation:
    def test_train_steps_computed_from_defaults(self):
        """Without epoch overrides, train.steps = defaults.epochs * spe."""
        from overrides_translator import steps_per_epoch as spe_fn
        job = _make_job("sdxl")
        spec = job.model_spec
        proc = _proc(_generate(job, img_count=10))
        # base batch=1, accum=1, num_repeats=1 → spe=10; defaults.epochs=100
        expected_steps = spec.defaults.epochs * spe_fn(10, 1, 1)
        assert proc["train"]["steps"] == expected_steps

    def test_save_every_computed_from_defaults(self):
        from overrides_translator import steps_per_epoch as spe_fn
        job = _make_job("sdxl")
        spec = job.model_spec
        proc = _proc(_generate(job, img_count=10))
        expected_save = spec.defaults.save_every_n_epochs * spe_fn(10, 1, 1)
        assert proc["save"]["save_every"] == expected_save

    def test_epoch_override_wins(self):
        from overrides_translator import steps_per_epoch as spe_fn
        job = _make_job("sdxl", config_overrides={"epochs": 50})
        proc = _proc(_generate(job, img_count=10))
        expected = 50 * spe_fn(10, 1, 1)
        assert proc["train"]["steps"] == expected

    def test_save_every_override_wins(self):
        from overrides_translator import steps_per_epoch as spe_fn
        job = _make_job("sdxl", config_overrides={"save_every_n_epochs": 3})
        proc = _proc(_generate(job, img_count=10))
        expected = 3 * spe_fn(10, 1, 1)
        assert proc["save"]["save_every"] == expected


# ---------------------------------------------------------------------------
# Deep-merge: override wins, base preserved
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_override_rank_wins_over_base(self):
        job = _make_job("sdxl", config_overrides={"adapter.rank": 64})
        proc = _proc(_generate(job))
        assert proc["network"]["linear"] == 64
        assert proc["network"]["linear_alpha"] == 64

    def test_override_lr_wins(self):
        job = _make_job("sdxl", config_overrides={"optimizer.lr": 5e-5})
        proc = _proc(_generate(job))
        assert proc["train"]["lr"] == pytest.approx(5e-5)

    def test_base_fields_preserved_when_no_override(self):
        """Un-overridden base fields should survive the merge."""
        proc = _proc(_generate(_make_job("sdxl")))
        assert "optimizer" in proc["train"] or "optimizer" in str(proc["train"])
        assert proc["train"].get("batch_size", 1) >= 1

    def test_override_batch_size(self):
        job = _make_job("sdxl", config_overrides={"micro_batch_size_per_gpu": 4})
        proc = _proc(_generate(job, img_count=10))
        assert proc["train"]["batch_size"] == 4


# ---------------------------------------------------------------------------
# N1 — YAML-hostile trigger_word values produce valid config + literal strings
# ---------------------------------------------------------------------------

YAML_HOSTILE_TRIGGERS = [
    ('sks "art" style', 'sks "art" style'),   # embedded quotes
    ('123', '123'),                             # all-digit → YAML might read as int
    ('yes', 'yes'),                             # YAML boolean
    ('colon:trigger', 'colon:trigger'),         # colon in value
    ('-leading-dash', '-leading-dash'),         # leading hyphen (YAML list item)
]


class TestN1YamlHostileTriggerWords:
    @pytest.mark.parametrize("trigger,expected", YAML_HOSTILE_TRIGGERS)
    def test_trigger_word_is_literal_string(self, trigger, expected):
        """Trigger word must survive round-trip as a literal Python str,
        immune to YAML type-coercion (N1 — post-parse assignment)."""
        job = _make_job("sdxl", trigger_word=trigger)
        proc = _proc(_generate(job))
        # Must be exactly the raw string, not coerced to int/bool/None
        tw = proc["datasets"][0]["trigger_word"]
        assert isinstance(tw, str), (
            f"trigger_word '{trigger}' was coerced to {type(tw).__name__}"
        )
        assert tw == expected

    @pytest.mark.parametrize("trigger,expected", YAML_HOSTILE_TRIGGERS)
    def test_no_trigger_word_placeholder_in_template(self, trigger, expected):
        """The config must not contain a bare {trigger_word} placeholder."""
        job = _make_job("sdxl", trigger_word=trigger)
        from yaml_generator import generate_config
        out_path = generate_config(job, 10)
        raw = out_path.read_text()
        assert "{trigger_word}" not in raw
        assert "{caption_prefix}" not in raw
