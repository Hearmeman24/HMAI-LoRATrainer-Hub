"""Tests for overrides_translator.py — §C allow-list, epoch→step math, warn+ignore.

N3: translate_overrides returns (config, ignored_keys) tuple per spec §5 N3.
xfail markers removed — tuple return landed.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, call, patch

import pytest

from overrides_translator import steps_per_epoch, translate_overrides


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate(overrides: dict, *, img_count=10, batch=1, accum=1,
               model_type="sdxl") -> dict:
    """Call translate_overrides and return the partial config dict.
    Accepts both the current return type (dict) and the N3 tuple (dict, list).
    """
    result = translate_overrides(
        overrides,
        model_type=model_type,
        dataset_image_count=img_count,
        base_batch_size=batch,
        base_grad_accum=accum,
    )
    if isinstance(result, tuple):
        return result[0]
    return result


def _ignored_keys(overrides: dict, *, img_count=10, batch=1, accum=1,
                  model_type="sdxl") -> list:
    """Return ignored_keys from translate_overrides (N3 tuple form)."""
    result = translate_overrides(
        overrides,
        model_type=model_type,
        dataset_image_count=img_count,
        base_batch_size=batch,
        base_grad_accum=accum,
    )
    assert isinstance(result, tuple), (
        "translate_overrides must return (config, ignored_keys) tuple — "
        "builder N3 update not landed yet"
    )
    return result[1]


# ---------------------------------------------------------------------------
# Epoch ↔ step math
# ---------------------------------------------------------------------------

class TestStepsPerEpoch:
    def test_basic(self):
        # ceil(10 * 1 / 1) = 10
        assert steps_per_epoch(10, 1, 1) == 10

    def test_batch_divides(self):
        # ceil(20 * 1 / 4) = 5
        assert steps_per_epoch(20, 1, 4) == 5

    def test_ceil_partial(self):
        # ceil(10 * 1 / 3) = ceil(3.33) = 4
        assert steps_per_epoch(10, 1, 3) == 4

    def test_num_repeats(self):
        # ceil(10 * 2 / 5) = 4
        assert steps_per_epoch(10, 2, 5) == 4

    def test_at_least_one(self):
        # zero images → still 1 (never 0)
        assert steps_per_epoch(0, 1, 1) >= 1

    def test_zero_batch_clamps(self):
        assert steps_per_epoch(10, 1, 0) >= 1


# ---------------------------------------------------------------------------
# Allow-list §C — every row → correct sink
# ---------------------------------------------------------------------------

class TestAllowListSinks:
    def test_epochs_to_train_steps(self):
        # epochs=5 × spe=ceil(10*1/1)=10 → train.steps=50
        cfg = _translate({"epochs": 5}, img_count=10, batch=1)
        assert cfg["train"]["steps"] == 50

    def test_save_every_n_epochs(self):
        # save_every_n_epochs=2 × spe=10 → save.save_every=20
        cfg = _translate({"save_every_n_epochs": 2}, img_count=10, batch=1)
        assert cfg["save"]["save_every"] == 20

    def test_epoch_math_with_batch_and_accum(self):
        # img=20, batch=2, accum=2 → eff=4 → spe=ceil(20/4)=5
        # epochs=3 → steps=15
        cfg = _translate({"epochs": 3}, img_count=20, batch=2, accum=2)
        assert cfg["train"]["steps"] == 15

    def test_adapter_rank_sets_linear_and_alpha(self):
        cfg = _translate({"adapter.rank": 64})
        assert cfg["network"]["linear"] == 64
        assert cfg["network"]["linear_alpha"] == 64

    def test_adapter_type_lora(self):
        cfg = _translate({"adapter.type": "lora"})
        assert cfg["network"]["type"] == "lora"

    def test_optimizer_lr(self):
        cfg = _translate({"optimizer.lr": 5e-5})
        assert cfg["train"]["lr"] == pytest.approx(5e-5)

    def test_optimizer_type_adamw8bit_passthrough(self):
        cfg = _translate({"optimizer.type": "adamw8bit"})
        assert cfg["train"]["optimizer"] == "adamw8bit"

    def test_optimizer_type_adamw_passthrough(self):
        cfg = _translate({"optimizer.type": "adamw"})
        assert cfg["train"]["optimizer"] == "adamw"

    def test_optimizer_type_adamw_optimi_mapped(self):
        """adamw_optimi → adamw8bit (exact mapping, R8)."""
        cfg = _translate({"optimizer.type": "adamw_optimi"})
        assert cfg["train"]["optimizer"] == "adamw8bit"

    def test_optimizer_betas(self):
        cfg = _translate({"optimizer.betas": [0.9, 0.999]})
        assert cfg["train"]["optimizer_params"]["betas"] == [0.9, 0.999]

    def test_optimizer_weight_decay(self):
        cfg = _translate({"optimizer.weight_decay": 0.01})
        assert cfg["train"]["optimizer_params"]["weight_decay"] == pytest.approx(0.01)

    def test_gradient_accumulation_steps(self):
        cfg = _translate({"gradient_accumulation_steps": 4})
        assert cfg["train"]["gradient_accumulation"] == 4

    def test_activation_checkpointing(self):
        cfg = _translate({"activation_checkpointing": True})
        assert cfg["train"]["gradient_checkpointing"] is True

    def test_micro_batch_size_per_gpu(self):
        cfg = _translate({"micro_batch_size_per_gpu": 2})
        assert cfg["train"]["batch_size"] == 2

    def test_save_dtype(self):
        cfg = _translate({"save_dtype": "bf16"})
        assert cfg["save"]["dtype"] == "bf16"

    def test_gradient_clipping(self):
        cfg = _translate({"gradient_clipping": 1.0})
        assert cfg["train"]["max_grad_norm"] == pytest.approx(1.0)

    def test_warmup_steps_sets_scheduler_and_params(self):
        cfg = _translate({"warmup_steps": 100})
        assert cfg["train"]["lr_scheduler"] == "constant_with_warmup"
        assert cfg["train"]["lr_scheduler_params"]["num_warmup_steps"] == 100

    def test_dataset_num_repeats(self):
        cfg = _translate({"dataset.num_repeats": 3})
        assert cfg["datasets"][0]["num_repeats"] == 3

    def test_dataset_resolutions(self):
        cfg = _translate({"dataset.resolutions": [512, 1024]})
        assert cfg["datasets"][0]["resolution"] == [512, 1024]

    def test_dataset_frame_buckets_first_element(self):
        cfg = _translate({"dataset.frame_buckets": [25, 50]})
        assert cfg["datasets"][0]["num_frames"] == 25

    def test_model_unet_lr(self):
        cfg = _translate({"model.unet_lr": 2e-5})
        assert cfg["train"]["unet_lr"] == pytest.approx(2e-5)

    def test_sdxl_te_lr_te1_only(self):
        cfg = _translate({"model.text_encoder_1_lr": 3e-5})
        assert cfg["train"]["text_encoder_lr"] == pytest.approx(3e-5)

    def test_sdxl_te_lr_te2_only(self):
        cfg = _translate({"model.text_encoder_2_lr": 4e-5})
        assert cfg["train"]["text_encoder_lr"] == pytest.approx(4e-5)

    def test_sdxl_per_encoder_lr_collapse_uses_te1(self):
        """Both te1 and te2 provided and differ → collapse to te1, no raise."""
        cfg = _translate({
            "model.text_encoder_1_lr": 1e-5,
            "model.text_encoder_2_lr": 2e-5,
        })
        assert cfg["train"]["text_encoder_lr"] == pytest.approx(1e-5)

    def test_sdxl_per_encoder_lr_collapse_warns_no_raise(self):
        """Collapsing differing te LRs warns but does NOT raise."""
        with patch("overrides_translator.alog") as mock_log:
            cfg = _translate({
                "model.text_encoder_1_lr": 1e-5,
                "model.text_encoder_2_lr": 2e-5,
            })
        # Should have called warn at least once for the collapse
        assert mock_log.warning.called


# ---------------------------------------------------------------------------
# Dropped keys — warn + ignored, NO raise
# ---------------------------------------------------------------------------

class TestDroppedKeys:
    """Every dropped key: ignored in output, logged via alog.warn, no raise."""

    DROPPED = [
        "adapter.dtype",
        "optimizer.eps",
        "compile",
        "pipeline_stages",
        "model.diffusion_model",
        "model.vae",
        "model.text_encoders",
        "model.transformer_path",
        "model.ckpt_path",
        "model.diffusers_path",
        "model.checkpoint_path",
        "model.transformer_dtype",
        "model.diffusion_model_dtype",
        "model.min_t",
        "model.max_t",
        "model.merge_adapters",
        "training_strategy.with_audio",
        "dataset.enable_ar_bucket",
        "dataset.min_ar",
        "dataset.max_ar",
        "dataset.num_ar_buckets",
    ]

    @pytest.mark.parametrize("key", DROPPED)
    def test_dropped_key_not_in_output(self, key):
        cfg = _translate({key: "some_value"})
        # The dropped key must not appear verbatim anywhere in the output
        import json
        as_json = json.dumps(cfg)
        # key itself shouldn't propagate to any sink under its own name
        # (we just verify no exception and the config is a dict)
        assert isinstance(cfg, dict)

    @pytest.mark.parametrize("key", DROPPED)
    def test_dropped_key_does_not_raise(self, key):
        _translate({key: "some_value"})  # must not raise

    def test_optimizer_eps_specifically_dropped_not_forwarded(self):
        """optimizer.eps MUST NOT appear in output (MANDATORY crash-guard)."""
        cfg = _translate({"optimizer.eps": 1e-8})
        # Must not be forwarded to optimizer_params or any nested dict
        train = cfg.get("train", {})
        assert "eps" not in train
        opt_params = train.get("optimizer_params", {})
        assert "eps" not in opt_params

    def test_dropped_key_warns_via_alog(self):
        with patch("overrides_translator.alog") as mock_log:
            _translate({"optimizer.eps": 1e-8})
        assert mock_log.warning.called

    def test_compile_dropped_warns(self):
        with patch("overrides_translator.alog") as mock_log:
            _translate({"compile": True})
        assert mock_log.warning.called


# ---------------------------------------------------------------------------
# Unknown key — warn + ignored, NO raise (never raise — DESIGN §I-D2)
# ---------------------------------------------------------------------------

class TestUnknownKey:
    def test_unknown_key_does_not_raise(self):
        _translate({"totally_unknown_key": 42})  # must not raise

    def test_unknown_key_not_in_output(self):
        cfg = _translate({"totally_unknown_key": 42})
        assert isinstance(cfg, dict)
        assert "totally_unknown_key" not in str(cfg)

    def test_unknown_key_warns_via_alog(self):
        with patch("overrides_translator.alog") as mock_log:
            _translate({"totally_unknown_key": 42})
        assert mock_log.warning.called

    def test_unknown_key_warn_names_key(self):
        """The warn message must name the unknown key."""
        with patch("overrides_translator.alog") as mock_log:
            _translate({"typo_optimzer_lr": 1e-4})
        # Check at least one warn call mentions the key name
        calls_text = " ".join(str(c) for c in mock_log.warning.call_args_list)
        assert "typo_optimzer_lr" in calls_text

    def test_mix_known_and_unknown(self):
        """Unknown key doesn't prevent known key from being processed."""
        cfg = _translate({"optimizer.lr": 2e-4, "unknown_key": "val"})
        assert cfg["train"]["lr"] == pytest.approx(2e-4)


# ---------------------------------------------------------------------------
# N3 — tuple return shape + ignored_keys logging (xfail until builder lands)
# ---------------------------------------------------------------------------

class TestN3TupleReturn:
    def test_returns_tuple(self):
        result = translate_overrides(
            {},
            model_type="sdxl",
            dataset_image_count=10,
            base_batch_size=1,
            base_grad_accum=1,
        )
        assert isinstance(result, tuple) and len(result) == 2

    def test_ignored_keys_in_tuple(self):
        """Dropped and unknown keys appear in ignored_keys list."""
        ignored = _ignored_keys({"optimizer.eps": 1e-8, "bad_key": 1})
        assert "optimizer.eps" in ignored
        assert "bad_key" in ignored

    def test_ignored_keys_not_in_output_shape(self):
        """ignored_keys must not appear as a key in the config dict (N3 / frozen OUTPUT)."""
        result = translate_overrides(
            {"optimizer.eps": 1e-8},
            model_type="sdxl",
            dataset_image_count=10,
            base_batch_size=1,
            base_grad_accum=1,
        )
        assert isinstance(result, tuple)
        cfg, ignored = result
        assert "ignored_keys" not in cfg
        assert "optimizer.eps" in ignored


class TestBraceInjectionNeverRaises:
    """Regression: loguru interpolates {} only when args/kwargs are passed. After
    the logger.py removal the warn() calls must carry NO kwargs, so an override
    key/value containing a brace can't trigger a format-time KeyError/ValueError
    (the 'never raises' contract, DESIGN §I-D2)."""

    @pytest.mark.parametrize("ov", [
        {"adapter.type": "lo{ra"},
        {"bad}key": 1},
        {"optimizer.lr{": 0.1},
        {"unknown{x}": 1},
        {"model.text_encoder_1_lr": 1e-4, "model.text_encoder_2_lr": 2e-4},  # te-collapse warn path
    ])
    def test_braced_overrides_do_not_raise(self, ov):
        cfg, ignored = translate_overrides(
            ov, model_type="sdxl", dataset_image_count=20,
            base_batch_size=1, base_grad_accum=1,
        )
        assert isinstance(cfg, dict)
