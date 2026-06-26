"""Tests for trainer.py — AI-Toolkit single subprocess + step-based watcher.

Subprocess, S3 upload, and webhook are mocked. Covers: filename re-key (step +
bare final, high/low), prune-race (renamed copy survives original deletion),
success==rc0 AND >=1 safetensors, rc0-with-no-output==failure, OUTPUT shape.
"""

import io
import json
import struct
from unittest.mock import MagicMock, patch

import pytest

import config as cfg
from config import AITK_JOB_NAME, validate_payload
from log_parser import TrainingProgress


def _write_safetensors(path, n_bytes: int = 16) -> None:
    """Write a minimal VALID safetensors file: 8-byte LE header length, the JSON
    header declaring one tensor of n_bytes, then n_bytes of data."""
    header = {"t": {"dtype": "F32", "shape": [n_bytes // 4 or 1],
                    "data_offsets": [0, n_bytes]}}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * n_bytes)


def _write_truncated_safetensors(path) -> None:
    """Write a safetensors header that PROMISES more tensor bytes than exist on
    disk (mid-write truncation) — _is_complete_safetensors must reject it."""
    header = {"t": {"dtype": "F32", "shape": [1000],
                    "data_offsets": [0, 4000]}}
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(b"\x00" * 100)  # only 100 of the promised 4000 bytes


def _make_job(tmp_path, model_type, **extra):
    cfg.JOBS_DIR = tmp_path / "jobs"
    payload = {
        "model_type": model_type,
        "dataset_zip_url": "https://example.com/data.zip",
        "trigger_word": "testword",
        "job_id": "test-job",
    }
    payload.update(extra)
    job = validate_payload(payload)
    for d in (job.configs_dir, job.output_dir, job.dataset_dir, job.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return job


@pytest.fixture(autouse=True)
def _restore_paths():
    orig = cfg.JOBS_DIR
    yield
    cfg.JOBS_DIR = orig


# ---------------------------------------------------------------------------
# Filename classification + epoch re-key
# ---------------------------------------------------------------------------

class TestClassify:
    def test_step_no_variant(self):
        from trainer import _classify
        assert _classify(f"{AITK_JOB_NAME}_000000250.safetensors") == ("step", 250, None)

    def test_step_high_noise(self):
        from trainer import _classify
        assert _classify(f"{AITK_JOB_NAME}_000001000_high_noise.safetensors") == ("step", 1000, "high")

    def test_step_low_noise(self):
        from trainer import _classify
        assert _classify(f"{AITK_JOB_NAME}_000000500_low_noise.safetensors") == ("step", 500, "low")

    def test_bare_final(self):
        from trainer import _classify
        assert _classify(f"{AITK_JOB_NAME}.safetensors") == ("bare", None, None)

    def test_bare_final_high(self):
        from trainer import _classify
        assert _classify(f"{AITK_JOB_NAME}_high_noise.safetensors") == ("bare", None, "high")

    def test_unrelated_file_ignored(self):
        from trainer import _classify
        assert _classify("some_other_lora.safetensors") is None
        assert _classify(f"{AITK_JOB_NAME}_00025.safetensors") is None  # not 9 digits

    def test_epoch_for_step(self):
        from trainer import _epoch_for_step
        assert _epoch_for_step(250, 50) == 5
        assert _epoch_for_step(1000, 50) == 20
        assert _epoch_for_step(10, 50) == 1   # rounds to >=1
        assert _epoch_for_step(250, 0) == 0   # guard


# ---------------------------------------------------------------------------
# _final_scan: re-key into _uploads/ with the frozen rename_output name
# ---------------------------------------------------------------------------

class TestFinalScan:
    @patch("trainer.upload_and_presign", return_value=None)
    def test_step_and_bare_rekey(self, _up, tmp_path):
        from trainer import _final_scan
        job = _make_job(tmp_path, "sdxl")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        _write_safetensors(save_dir / f"{AITK_JOB_NAME}_000000250.safetensors")
        _write_safetensors(save_dir / f"{AITK_JOB_NAME}.safetensors")  # bare final
        uploads = job.output_dir / "_uploads"
        progress = TrainingProgress()

        _final_scan(save_dir, job, progress, steps_per_epoch=50, total_epochs=10, uploads_dir=uploads)

        names = sorted(p.name for p in uploads.glob("*.safetensors"))
        # step 250 / spe 50 -> epoch5 ; bare -> total_epochs=10
        assert names == ["testword_sdxl_epoch10.safetensors", "testword_sdxl_epoch5.safetensors"]

    @patch("trainer.upload_and_presign", return_value=None)
    def test_wan_variant_suffix(self, _up, tmp_path):
        from trainer import _final_scan
        job = _make_job(tmp_path, "wan2.2", noise_variant="high")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        _write_safetensors(save_dir / f"{AITK_JOB_NAME}_000000500_high_noise.safetensors")
        uploads = job.output_dir / "_uploads"
        progress = TrainingProgress()

        _final_scan(save_dir, job, progress, steps_per_epoch=100, total_epochs=5, uploads_dir=uploads)

        names = [p.name for p in uploads.glob("*.safetensors")]
        # variant comes from job.noise_variant (the filename suffix → rename_output)
        assert names == ["testword_wan2.2high_epoch5.safetensors"]

    @patch("trainer.upload_and_presign", return_value=None)
    def test_dedup_against_existing_upload(self, _up, tmp_path):
        from trainer import _final_scan
        job = _make_job(tmp_path, "sdxl")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        _write_safetensors(save_dir / f"{AITK_JOB_NAME}_000000250.safetensors")
        uploads = job.output_dir / "_uploads"
        uploads.mkdir()
        # pre-existing renamed copy (already published by the live watcher)
        (uploads / "testword_sdxl_epoch5.safetensors").write_bytes(b"a")
        progress = TrainingProgress()

        _final_scan(save_dir, job, progress, steps_per_epoch=50, total_epochs=10, uploads_dir=uploads)
        # nothing new published for the already-present (deduped) file
        assert progress.ready_loras == []


# ---------------------------------------------------------------------------
# Prune-race: renamed copy lives in a sibling dir outside the prune glob
# ---------------------------------------------------------------------------

class TestPruneRace:
    @patch("trainer.upload_and_presign", return_value=None)
    def test_copy_survives_original_deletion(self, _up, tmp_path):
        from trainer import _publish
        job = _make_job(tmp_path, "sdxl")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        src = save_dir / f"{AITK_JOB_NAME}_000000250.safetensors"
        src.write_bytes(b"weights")
        uploads = job.output_dir / "_uploads"
        progress = TrainingProgress()

        _publish(src, epoch=5, variant=None, job=job, progress=progress, uploads_dir=uploads)
        renamed = uploads / "testword_sdxl_epoch5.safetensors"
        assert renamed.exists()
        # AI-Toolkit prunes the original (different glob {name}_*); copy is untouched
        src.unlink()
        assert renamed.exists() and renamed.read_bytes() == b"weights"
        # uploads dir is NOT under the prune glob save_dir
        assert uploads.parent == save_dir.parent
        assert uploads != save_dir


# ---------------------------------------------------------------------------
# run_training success semantics (subprocess mocked; checkpoints pre-staged)
# ---------------------------------------------------------------------------

def _mock_proc(rc, stdout="", stderr="", on_wait=None):
    # stdout/stderr are read char-by-char via .read(1) (F5 \r/\n splitter), so use
    # real text streams, not list iterators.
    proc = MagicMock()
    proc.stdout = io.StringIO(stdout)
    proc.stderr = io.StringIO(stderr)
    proc.returncode = rc

    def _wait(timeout=None):
        if on_wait:
            on_wait()
        return rc

    proc.wait.side_effect = _wait
    return proc


class TestRunTraining:
    def _write_config(self, job, steps=500):
        (job.configs_dir / "config.yml").write_text(
            "job: extension\n"
            "config:\n"
            "  name: aitk_lora\n"
            "  process:\n"
            "    - train: {steps: %d, batch_size: 1, gradient_accumulation: 1}\n"
            "      datasets: [{num_repeats: 1}]\n" % steps
        )
        # 10 media so spe = ceil(10/1)=10 -> total_epochs = 500/10 = 50
        for i in range(10):
            (job.dataset_dir / f"img{i}.png").write_bytes(b"x")

    @patch("trainer.upload_and_presign", return_value={"presigned_url": "https://s3/x"})
    @patch("trainer.subprocess.Popen")
    def test_success_rc0_with_output(self, mock_popen, _up, tmp_path):
        from trainer import run_training
        job = _make_job(tmp_path, "sdxl")
        self._write_config(job)
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)

        # subprocess "produces" a bare final checkpoint by the time it exits
        def produce():
            _write_safetensors(save_dir / f"{AITK_JOB_NAME}.safetensors")

        mock_popen.return_value = _mock_proc(0, on_wait=produce)

        result = run_training(job)
        assert result.ok is True
        assert len(result.output_files) == 1
        f = result.output_files[0]
        assert set(f) >= {"filename", "local_path", "noise_variant"}
        assert f["filename"] == "testword_sdxl_epoch50.safetensors"  # bare -> total_epochs
        # cwd is AI-Toolkit dir, cmd is `python run.py config.yml`
        _, kwargs = mock_popen.call_args
        assert kwargs["cwd"] == str(cfg.AI_TOOLKIT_DIR)
        assert mock_popen.call_args[0][0][-1].endswith("config.yml")

    @patch("trainer.upload_and_presign", return_value=None)
    @patch("trainer.subprocess.Popen")
    def test_rc0_no_output_is_failure(self, mock_popen, _up, tmp_path):
        from trainer import run_training
        job = _make_job(tmp_path, "sdxl")
        self._write_config(job)
        (job.output_dir / AITK_JOB_NAME).mkdir(parents=True)
        mock_popen.return_value = _mock_proc(0)  # exits clean, writes nothing

        result = run_training(job)
        assert result.ok is False
        assert result.error_type == "TRAINING"
        assert "no .safetensors" in result.error

    @patch("trainer.upload_and_presign", return_value=None)
    @patch("trainer.subprocess.Popen")
    def test_rc_nonzero_is_failure(self, mock_popen, _up, tmp_path):
        from trainer import run_training
        job = _make_job(tmp_path, "sdxl")
        self._write_config(job)
        (job.output_dir / AITK_JOB_NAME).mkdir(parents=True)
        mock_popen.return_value = _mock_proc(1, stderr="CUDA OOM\n")

        result = run_training(job)
        assert result.ok is False
        assert result.error_type == "TRAINING"
        assert "exit code 1" in result.error


# ---------------------------------------------------------------------------
# N2 write-completion gate
# ---------------------------------------------------------------------------

class TestWriteCompletionGate:
    def test_complete_header_accepted(self, tmp_path):
        from trainer import _is_complete_safetensors
        p = tmp_path / "ok.safetensors"
        _write_safetensors(p, n_bytes=64)
        assert _is_complete_safetensors(p) is True

    def test_truncated_header_rejected(self, tmp_path):
        from trainer import _is_complete_safetensors
        p = tmp_path / "trunc.safetensors"
        _write_truncated_safetensors(p)  # header promises 4000 data bytes, only 100 written
        assert _is_complete_safetensors(p) is False

    def test_too_small_for_length_prefix(self, tmp_path):
        from trainer import _is_complete_safetensors
        p = tmp_path / "tiny.safetensors"
        p.write_bytes(b"\x00\x00")  # < 8 bytes
        assert _is_complete_safetensors(p) is False

    def test_garbage_rejected(self, tmp_path):
        from trainer import _is_complete_safetensors
        p = tmp_path / "junk.safetensors"
        p.write_bytes(b"final")  # old test value — must NOT validate
        assert _is_complete_safetensors(p) is False

    @patch("trainer.upload_and_presign", return_value=None)
    def test_scan_skips_unstable_then_publishes_when_stable(self, _up, tmp_path):
        """A file whose size changes between polls is NOT published; once stable
        (and complete) it is. Drives _scan_once directly across two polls."""
        from trainer import _scan_once
        job = _make_job(tmp_path, "sdxl")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        cp = save_dir / f"{AITK_JOB_NAME}_000000250.safetensors"
        _write_safetensors(cp, n_bytes=64)
        uploads = job.output_dir / "_uploads"
        progress = TrainingProgress()
        seen, sizes = set(), {}

        # poll 1: first sighting → size recorded, NOT stable yet → no publish
        _scan_once(save_dir, job, progress, seen, sizes, 50, 10, uploads)
        assert not list(uploads.glob("*.safetensors")) if uploads.exists() else True
        assert progress.ready_loras == []

        # poll 2: same size → stable + complete header → published
        _scan_once(save_dir, job, progress, seen, sizes, 50, 10, uploads)
        assert (uploads / "testword_sdxl_epoch5.safetensors").exists()
        assert len(progress.ready_loras) == 1

    @patch("trainer.upload_and_presign", return_value=None)
    def test_scan_keeps_truncated_pending(self, _up, tmp_path):
        """A stable-size but truncated-header file is NOT published (kept pending),
        guarding against a corrupt deliverable counted as success."""
        from trainer import _scan_once
        job = _make_job(tmp_path, "sdxl")
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)
        cp = save_dir / f"{AITK_JOB_NAME}_000000250.safetensors"
        _write_truncated_safetensors(cp)
        uploads = job.output_dir / "_uploads"
        progress = TrainingProgress()
        seen, sizes = set(), {}

        _scan_once(save_dir, job, progress, seen, sizes, 50, 10, uploads)  # poll 1
        _scan_once(save_dir, job, progress, seen, sizes, 50, 10, uploads)  # poll 2 (stable but truncated)
        assert not (uploads.exists() and list(uploads.glob("*.safetensors")))
        assert progress.ready_loras == []
        assert cp.name not in seen  # stays pending for a later poll

    @patch("trainer.upload_and_presign", return_value={"presigned_url": "u"})
    @patch("trainer.subprocess.Popen")
    def test_final_sweep_skips_truncated_final(self, mock_popen, _up, tmp_path):
        """rc0 but the only output is a truncated final file → NOT success (D6)."""
        from trainer import run_training
        job = _make_job(tmp_path, "sdxl")
        TestRunTraining()._write_config(job)
        save_dir = job.output_dir / AITK_JOB_NAME
        save_dir.mkdir(parents=True)

        def produce():
            _write_truncated_safetensors(save_dir / f"{AITK_JOB_NAME}.safetensors")

        mock_popen.return_value = _mock_proc(0, on_wait=produce)
        result = run_training(job)
        assert result.ok is False
        assert "no .safetensors" in result.error


# ---------------------------------------------------------------------------
# F5: stderr \r-split (tqdm live bar) feeds progress
# ---------------------------------------------------------------------------

class TestStreamRSplit:
    def test_iter_lines_splits_on_cr_and_lf(self):
        import io as _io
        from trainer import _iter_lines
        stream = _io.StringIO("a\rb\nc\rd")
        assert list(_iter_lines(stream)) == ["a", "b", "c", "d"]

    @patch("trainer.subprocess.Popen")
    def test_cr_progress_parsed_from_stderr(self, mock_popen, tmp_path):
        """tqdm renders the bar on stderr as \\r-terminated frames with no newline.
        _stream_output must split on \\r so parse_line sees 'N/total ... loss'."""
        from trainer import _stream_output
        # three carriage-return frames, no trailing newline until the last
        stderr = ("aitk_lora:  5%| | 50/1000 [..] lr: 1.0e-04 loss: 2.000e-01\r"
                  "aitk_lora: 25%| | 250/1000 [..] lr: 1.0e-04 loss: 1.000e-01\r"
                  "aitk_lora: 50%| | 500/1000 [..] lr: 1.0e-04 loss: 5.000e-02\r")
        proc = _mock_proc(0, stdout="", stderr=stderr)
        progress = TrainingProgress(total_steps=1000)

        _stream_output(proc, progress, "sdxl")
        # the LAST \r-frame must have been parsed (step 500), not stalled at 0
        assert progress.step == 500
        assert progress.total_steps == 1000
