"""Breaker code-attack (task #13) — NEW mutations against the REAL shipped code.

Distinct from the staged stubs: these attack the actual trainer/translator
symbols (not placeholder names) with sequences the design didn't enumerate.
Findings that go RED here are real bugs in shipped code, routed to the owning
slice. Isolated under tests/stubs/ (conftest keeps them out of the default run).

Run: RUN_BREAKER_STUBS=1 pytest tests/stubs/test_breaker_attack.py -v
"""

from __future__ import annotations

import re
import struct
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import config as cfg


def _valid_safetensors(path: Path, n_tensor_bytes: int = 64) -> None:
    header = json.dumps(
        {"w": {"dtype": "F16", "shape": [n_tensor_bytes // 2],
               "data_offsets": [0, n_tensor_bytes]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * n_tensor_bytes)


def _make_job(tmp_path, model_type="sdxl", noise_variant=None):
    cfg.JOBS_DIR = tmp_path / "jobs"
    raw = {
        "model_type": model_type,
        "dataset_zip_url": "https://x/data.zip",
        "trigger_word": "sks",
        "job_id": "job1",
    }
    if noise_variant is not None:
        raw["noise_variant"] = noise_variant
    job = cfg.validate_payload(raw)
    for d in (job.configs_dir, job.output_dir, job.logs_dir, job.dataset_dir):
        d.mkdir(parents=True, exist_ok=True)
    return job


# ===========================================================================
# ATTACK 1 — live watcher publishes the SAME epoch TWICE when a step checkpoint
# and the bare final map to the same epoch number.
#   _scan_once dedups by SOURCE name (`seen`), but a step file and the bare file
#   are different source names that rename to the SAME target when
#   _epoch_for_step(final_step) == total_epochs. Result: send_checkpoint_ready
#   fires twice for one epoch + the second copy2 overwrites the first.
#   (_final_scan dedups by TARGET and is correct — the bug is the live path.)
# ===========================================================================

class TestAttack1DuplicateEpochPublish:
    def test_step_and_bare_same_epoch_double_publish(self, tmp_path):
        import trainer
        job = _make_job(tmp_path)
        save_dir = job.output_dir / cfg.AITK_JOB_NAME
        save_dir.mkdir(parents=True, exist_ok=True)
        uploads_dir = job.output_dir.parent / cfg.UPLOADS_SUBDIR

        # spe=30, total_epochs=10. Final step 300 → epoch 10; bare → epoch 10.
        spe, total_epochs = 30, 10
        # Both a step-300 file and the bare final exist in save_dir.
        _valid_safetensors(save_dir / f"{cfg.AITK_JOB_NAME}_000000300.safetensors")
        _valid_safetensors(save_dir / f"{cfg.AITK_JOB_NAME}.safetensors")

        progress = __import__("log_parser").TrainingProgress(total_epochs=total_epochs)
        seen: set[str] = set()
        sizes: dict[str, int] = {}

        with patch("trainer.upload_and_presign", return_value=None):
            # Two polls so the N2 size-stability gate promotes both files.
            trainer._scan_once(save_dir, job, progress, seen, sizes, spe,
                               total_epochs, uploads_dir)
            trainer._scan_once(save_dir, job, progress, seen, sizes, spe,
                               total_epochs, uploads_dir)

        # Publishing is now observable via progress.ready_loras (one entry per
        # checkpoint _publish appended). Derive the epoch from each filename.
        published = [r["filename"] for r in progress.ready_loras]
        epochs_fired = [int(m.group(1))
                        for f in published
                        if (m := re.search(r"epoch(\d+)", f))]
        # Both files map to epoch 10 → the renamed target collides.
        assert epochs_fired.count(10) <= 1, (
            "BUG (Attack1): live watcher published epoch 10 TWICE "
            f"(step-300 file AND bare final both rename to the same target). "
            f"epochs published={epochs_fired}. _scan_once dedups by SOURCE name, not by "
            f"the renamed TARGET like _final_scan does — a duplicate publish + "
            f"overwriting upload for one epoch (contract drift)."
        )


# ===========================================================================
# ATTACK 2 — N2 gate: a file that GROWS then SHRINKS back to its first size.
#   _is_stable compares only prev==now. If poll1 sees size S, poll2 sees a grown
#   size, poll3 sees it back at S and stable, the header check is the only thing
#   standing between a transiently-corrupt file and an upload. Verify the header
#   check actually rejects a file that is size-stable but truncated.
# ===========================================================================

class TestAttack2N2GrowShrink:
    def test_size_stable_but_truncated_header_rejected(self, tmp_path):
        import trainer
        p = tmp_path / f"{cfg.AITK_JOB_NAME}_000000090.safetensors"
        # claims a 10000-byte header but only writes a few bytes → truncated
        p.write_bytes(struct.pack("<Q", 10_000) + b'{"w":')
        sizes = {p.name: p.stat().st_size}  # pretend prev poll saw this exact size
        # size is "stable" (prev==now) ...
        assert trainer._is_stable(p, sizes) is True
        # ... but the completion gate must still reject it (header doesn't parse).
        assert trainer._is_complete_safetensors(p) is False, (
            "a size-stable but truncated-header file must fail _is_complete_safetensors"
        )

    def test_first_sighting_never_stable(self, tmp_path):
        """First time a file is seen, prev size is unknown → must NOT be promoted
        (a file could be fully written on first sight by luck, but the gate must
        wait one poll to be safe). Guards the N2 'across two polls' contract."""
        import trainer
        p = tmp_path / f"{cfg.AITK_JOB_NAME}_000000090.safetensors"
        _valid_safetensors(p)
        sizes: dict[str, int] = {}
        assert trainer._is_stable(p, sizes) is False, (
            "first sighting must not be 'stable' — N2 requires size stable ACROSS "
            "two consecutive polls"
        )


# ===========================================================================
# ATTACK 3 — epoch math on degenerate datasets + rounding boundaries.
# ===========================================================================

class TestAttack3EpochMath:
    def test_one_image_dataset_no_div_by_zero(self):
        from overrides_translator import steps_per_epoch
        # 1 image, 1 repeat, batch 1 → spe must be >=1, never 0 / ZeroDivision.
        assert steps_per_epoch(1, 1, 1) == 1
        # zero images (empty dataset) must not crash the epoch math.
        assert steps_per_epoch(0, 1, 1) >= 1

    def test_epoch_for_step_zero_steps_per_epoch(self):
        import trainer
        # Defensive: spe=0 must not ZeroDivision (returns 0 epoch).
        assert trainer._epoch_for_step(100, 0) == 0

    def test_epoch_for_step_half_boundary_is_monotonic(self):
        """Banker's rounding (round-half-to-even) makes epoch tags non-monotonic
        at .5 boundaries: round(0.5)=0, round(1.5)=2, round(2.5)=2. Two adjacent
        half-step saves can collide or skip an epoch NUMBER. Cosmetic per design,
        but assert the sequence is at least non-decreasing as step increases."""
        import trainer
        spe = 30
        epochs = [trainer._epoch_for_step(s, spe) for s in range(0, 301, 15)]
        non_decreasing = all(b >= a for a, b in zip(epochs, epochs[1:]))
        assert non_decreasing, (
            f"epoch tags must be non-decreasing in step; banker's rounding broke it: "
            f"{epochs} — consider int(step/spe + 0.5) or floor-based mapping"
        )


# ===========================================================================
# ATTACK 4 — noise_variant gating holes (real validate_payload).
# ===========================================================================

class TestAttack4NoiseVariantGating:
    def test_wan_rejects_non_string_variant(self, tmp_path):
        _make_job  # touch fixture import path
        cfg.JOBS_DIR = tmp_path / "jobs"
        base = {"model_type": "wan2.2", "dataset_zip_url": "u", "trigger_word": "t"}
        for bad in (["high"], 1, {"v": "high"}, "HIGH", " high", "high "):
            with pytest.raises(cfg.PayloadError):
                cfg.validate_payload({**base, "noise_variant": bad})

    def test_wan_accepts_exact_enum(self, tmp_path):
        cfg.JOBS_DIR = tmp_path / "jobs"
        base = {"model_type": "wan2.2", "dataset_zip_url": "u", "trigger_word": "t"}
        for ok in ("high", "low"):
            job = cfg.validate_payload({**base, "noise_variant": ok})
            assert job.noise_variant == ok

    def test_non_wan_rejects_any_variant_including_empty(self, tmp_path):
        cfg.JOBS_DIR = tmp_path / "jobs"
        base = {"model_type": "sdxl", "dataset_zip_url": "u", "trigger_word": "t"}
        # a non-wan request carrying noise_variant at all is a contract violation.
        with pytest.raises(cfg.PayloadError):
            cfg.validate_payload({**base, "noise_variant": "high"})

    def test_non_wan_empty_string_variant(self, tmp_path):
        """Edge: sdxl with noise_variant='' — '' is not None so the gate fires.
        Documents the behavior (reject) so a future refactor can't silently let
        an empty variant slip through for a non-wan model."""
        cfg.JOBS_DIR = tmp_path / "jobs"
        base = {"model_type": "sdxl", "dataset_zip_url": "u", "trigger_word": "t"}
        with pytest.raises(cfg.PayloadError):
            cfg.validate_payload({**base, "noise_variant": ""})


# ===========================================================================
# ATTACK 5 — uploader/collect idempotency: a duplicate renamed file in uploads
# must not appear twice in the OUTPUT set.
# ===========================================================================

class TestAttack5OutputIdempotency:
    def test_collect_outputs_no_duplicate_filenames(self, tmp_path):
        import trainer
        job = _make_job(tmp_path)
        uploads_dir = job.output_dir.parent / cfg.UPLOADS_SUBDIR
        uploads_dir.mkdir(parents=True, exist_ok=True)
        # one renamed final file
        name = cfg.rename_output("sks", "sdxl", None, 10)
        _valid_safetensors(uploads_dir / name)
        out = trainer._collect_outputs(uploads_dir, job)
        names = [f["filename"] for f in out]
        assert len(names) == len(set(names)), f"duplicate filenames in OUTPUT: {names}"
        # OUTPUT contract: each entry has filename + noise_variant keys.
        for f in out:
            assert "filename" in f and "noise_variant" in f, (
                f"OUTPUT file entry missing frozen keys: {f}"
            )
