"""Tests for log_parser.py — regex parsing of AI-Toolkit tqdm/stderr lines."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from log_parser import TrainingProgress, parse_line, report_progress, should_report


# ---------------------------------------------------------------------------
# Primary path: STEP_LOSS_PATTERN (tqdm bar with N/total + loss postfix)
# ---------------------------------------------------------------------------

class TestParseTqdmBar:
    """AI-Toolkit canonical tqdm line (SOURCE_FINDINGS §2).

    Format: `{name}:  {pct}%|bar| {n}/{total} [elapsed<remaining, rate, lr: X loss: Y]`
    """

    def test_full_bar_extracts_step_and_loss(self):
        """Full rendered bar: step, total, loss all extracted in one pass."""
        p = TrainingProgress(total_steps=2000)
        line = "aitk_lora:  12%|█▎        | 250/2000 [01:23<09:45,  2.99it/s, lr: 1.0e-04 loss: 1.234e-01]"
        assert parse_line(line, p)
        assert p.step == 250
        assert p.loss == pytest.approx(1.234e-01, rel=1e-3)
        assert p.stage == "training"

    def test_percent_from_pre_set_total(self):
        """Percent uses caller-supplied total_steps, not bar denominator."""
        p = TrainingProgress(total_steps=2000)
        parse_line("aitk_lora:  12%|█▎        | 250/2000 [01:23<09:45, lr: 1.0e-04 loss: 1.234e-01]", p)
        assert p.percent == pytest.approx(12.5)

    def test_adopts_bar_total_when_not_pre_set(self):
        """When total_steps=0, adopt the bar's denominator."""
        p = TrainingProgress(total_steps=0)
        parse_line("aitk_lora:  25%|##        | 250/1000 [00:30<01:30, lr: 1.0e-04 loss: 5.0e-02]", p)
        assert p.total_steps == 1000
        assert p.step == 250
        assert p.percent == pytest.approx(25.0)

    def test_does_not_overwrite_pre_set_total(self):
        """A caller-supplied total_steps must not be clobbered by the bar."""
        p = TrainingProgress(total_steps=3000)
        parse_line("aitk_lora:  8%|#         | 250/2000 [00:10<02:00, lr: 1.0e-04 loss: 0.1]", p)
        assert p.total_steps == 3000  # pre-set wins

    def test_loss_colon_space_format(self):
        """Loss postfix uses `loss: value` (colon + space) per §2."""
        p = TrainingProgress()
        parse_line("job:   5%|          | 50/1000 [loss: 3.456e-01]", p)
        assert p.loss == pytest.approx(3.456e-01, rel=1e-3)

    def test_loss_equals_format(self):
        """STEP_LOSS_PATTERN handles `loss=value` too."""
        p = TrainingProgress()
        parse_line("job:   5%|          | 50/1000 [loss=2.0e-01]", p)
        assert p.loss == pytest.approx(0.2, rel=1e-3)

    def test_returns_true_on_match(self):
        p = TrainingProgress()
        result = parse_line("x:  1%|  | 10/1000 [lr: 1e-04 loss: 0.1]", p)
        assert result is True

    def test_f5_breaker_exact_line(self):
        """F5 contract: exact line from test_breaker_staged.TestF5CarriageReturnBar."""
        p = TrainingProgress(total_steps=0)
        line = "aitk_lora:  25%|##5       | 250/1000 [01:00<03:00, lr: 1.0e-04 loss: 1.234e-01]"
        updated = parse_line(line, p)
        assert updated
        assert p.step == 250
        assert p.loss == pytest.approx(1.234e-01, rel=1e-3)
        assert p.total_steps in (0, 1000)  # breaker contract: either 0 or 1000


# ---------------------------------------------------------------------------
# Fallback path: STEP_PATTERN (N/total without loss)
# ---------------------------------------------------------------------------

class TestParseTqdmBarNoLoss:
    """Bar lines rendered before the postfix is populated."""

    def test_step_extracted_without_loss(self):
        """N/total without postfix: step and total extracted, loss unchanged."""
        p = TrainingProgress(total_steps=500)
        assert parse_line("aitk_lora:   5%|          | 25/500 [00:05<01:40]", p)
        assert p.step == 25
        assert p.loss is None
        assert p.stage == "training"

    def test_percent_from_bar_when_no_total_preset(self):
        p = TrainingProgress(total_steps=0)
        parse_line("job: 10%|# | 100/1000 [00:10]", p)
        assert p.total_steps == 1000
        assert p.percent == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Standalone loss fallback: LOSS_PATTERN
# ---------------------------------------------------------------------------

class TestParseLossStandalone:
    def test_loss_equals(self):
        p = TrainingProgress()
        assert parse_line("loss=0.0543", p)
        assert p.loss == pytest.approx(0.0543)

    def test_avr_loss(self):
        p = TrainingProgress()
        assert parse_line("avr_loss=0.0543", p)
        assert p.loss == pytest.approx(0.0543)

    def test_loss_scientific(self):
        p = TrainingProgress()
        assert parse_line("loss: 1.23e-02", p)
        assert p.loss == pytest.approx(0.0123)

    def test_loss_colon_no_slash(self):
        """Standalone `loss: X` (no N/total) hits LOSS_PATTERN, not STEP_PATTERN."""
        p = TrainingProgress()
        assert parse_line("INFO train loss: 0.042", p)
        assert p.loss == pytest.approx(0.042)


# ---------------------------------------------------------------------------
# No-match path
# ---------------------------------------------------------------------------

class TestUnrelatedLine:
    def test_no_match_returns_false(self):
        p = TrainingProgress()
        assert not parse_line("Loading model weights...", p)

    def test_no_match_does_not_change_state(self):
        p = TrainingProgress()
        parse_line("Loading model weights...", p)
        assert p.step == 0
        assert p.loss is None
        assert p.stage == "initializing"

    def test_empty_line(self):
        p = TrainingProgress()
        assert not parse_line("", p)

    def test_carriage_return_only(self):
        """A bare \r (split artifact) must not crash or update."""
        p = TrainingProgress()
        assert not parse_line("\r", p)


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------

class TestThrottling:
    def test_reports_after_interval(self):
        p = TrainingProgress()
        p._last_report_time = time.time() - 15
        assert should_report(p) is True

    def test_throttled_immediately_after_report(self):
        p = TrainingProgress()
        p._last_report_time = time.time() - 15
        should_report(p)
        assert should_report(p) is False

    def test_not_yet_due(self):
        p = TrainingProgress()
        p._last_report_time = time.time() - 5  # only 5s, threshold is 10
        assert should_report(p) is False


# ---------------------------------------------------------------------------
# progress_dict contract (frozen interface, spec §B)
# ---------------------------------------------------------------------------

class TestProgressDict:
    def test_with_all_fields(self):
        p = TrainingProgress(
            step=10, total_steps=100, epoch=2, total_epochs=80,
            loss=0.05, percent=10.0, label="wan22_high",
        )
        d = p.progress_dict
        assert d["step"] == 10
        assert d["total_steps"] == 100
        assert d["epoch"] == 2
        assert d["total_epochs"] == 80
        assert d["loss"] == 0.05
        assert d["percent"] == 10.0
        assert d["label"] == "wan22_high"

    def test_loss_absent_when_none(self):
        p = TrainingProgress(step=10, total_steps=100, epoch=2, total_epochs=80)
        assert "loss" not in p.progress_dict

    def test_label_absent_when_empty(self):
        p = TrainingProgress(step=10, total_steps=100)
        assert "label" not in p.progress_dict

    def test_ready_loras_present_when_set(self):
        p = TrainingProgress(step=10, total_steps=100)
        p.ready_loras.append({"filename": "test.safetensors", "url": "https://example.com"})
        d = p.progress_dict
        assert len(d["ready_loras"]) == 1
        assert d["ready_loras"][0]["filename"] == "test.safetensors"

    def test_ready_loras_absent_when_empty(self):
        p = TrainingProgress(step=10, total_steps=100)
        assert "ready_loras" not in p.progress_dict


# ---------------------------------------------------------------------------
# report_progress
# ---------------------------------------------------------------------------

class TestReportProgress:
    def test_calls_runpod_when_job_id_set(self):
        mock_runpod = MagicMock()
        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            p = TrainingProgress(
                step=50, total_steps=100, epoch=3, total_epochs=80,
                loss=0.05, percent=50.0, job_id="test-job-123",
            )
            report_progress(p)
            mock_runpod.serverless.progress_update.assert_called_once()
            call_args = mock_runpod.serverless.progress_update.call_args
            assert call_args.args[0] == {"id": "test-job-123"}
            assert call_args.args[1]["step"] == 50
            assert call_args.args[1]["total_steps"] == 100

    def test_skips_runpod_when_no_job_id(self):
        p = TrainingProgress(step=50, total_steps=100, percent=50.0)
        report_progress(p)  # must not raise

    def test_does_not_crash_on_runpod_failure(self):
        mock_runpod = MagicMock()
        mock_runpod.serverless.progress_update.side_effect = Exception("connection error")
        with patch.dict("sys.modules", {"runpod": mock_runpod}):
            p = TrainingProgress(step=50, total_steps=100, percent=50.0, job_id="test-job")
            report_progress(p)  # must not raise


class TestProgressDenominatorAndEpoch:
    """Regression: a setup/caching tqdm bar must NOT clobber the pre-set
    total_steps, and the epoch must be synthesized from the step (no epoch line
    in AI-Toolkit's step-based output). Bug: total_steps=28 / percent=3792% /
    epoch=0 reported during a real run."""

    def test_caching_bar_does_not_clobber_preset_total(self):
        p = TrainingProgress(total_steps=4400, total_epochs=10, steps_per_epoch=440)
        # a caching/setup bar with a small denominator appears first
        parse_line("Caching latents: 100%|####| 28/28", p)
        assert p.total_steps == 4400  # NOT overwritten to 28

    def test_real_training_line_sets_step_and_synth_epoch(self):
        p = TrainingProgress(total_steps=4400, total_epochs=10, steps_per_epoch=440)
        parse_line("Caching latents: 100%|####| 28/28", p)
        parse_line(
            "aitk_lora:  24%|##| 1062/4400 [12:30<40:00, 1.5it/s, lr: 1.0e-04 loss: 5.6e-01]",
            p,
        )
        assert p.step == 1062
        assert p.total_steps == 4400
        assert p.epoch == 3            # ceil(1062/440)
        assert abs(p.percent - 24.1) < 0.3
        assert abs(p.loss - 0.56) < 1e-6

    def test_epoch_capped_at_total_epochs(self):
        p = TrainingProgress(total_steps=4400, total_epochs=10, steps_per_epoch=440)
        parse_line("aitk_lora: 100%|##| 4400/4400 [.., loss: 1.0e-01]", p)
        assert p.epoch == 10           # not 11

    def test_no_steps_per_epoch_leaves_epoch_untouched(self):
        # when steps_per_epoch is unknown (0), epoch is not synthesized
        p = TrainingProgress(total_steps=100)
        parse_line("foo 50/100 loss: 0.1", p)
        assert p.epoch == 0
