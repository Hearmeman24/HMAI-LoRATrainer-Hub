"""Failing-test stubs (RED until the slices implement the contracts).

Authored by the architect (task #15) → handed to the breaker (task #13) to
attack the real code. Each stub encodes the EXACT contract from DESIGN.md and
is expected to FAIL against current code, going green only when the relevant
slice lands:

  F5  -> LOGPARSER (#6) + TRAINER (#5): tqdm renders `N/total` only in the live
         bar on STDERR, `\\r`-terminated, no `\\n` until close. The reader MUST
         split on BOTH `\\r` and `\\n` or progress stalls at 0%. (DESIGN §B6.)
  F1  -> TRAINER (#5): AI-Toolkit's final `save()` emits a STEP-LESS bare
         `{AITK_JOB_NAME}[_noise].safetensors` (the best LoRA). The watcher /
         final-collect must capture it via a SECOND (bare) pattern and map it to
         epoch == total_epochs, renaming via rename_output into `_uploads/`.
         (DESIGN §B5 two-pattern + §B4 final-sweep.)
  N2  -> TRAINER (#5): AI-Toolkit's save_file writes IN-PLACE (non-atomic). The
         watcher must NOT copy/upload a checkpoint until its size is stable
         across two polls AND its safetensors header parses. (DESIGN §B5.)

These live under tests/stubs/ so they are DISJOINT from CONFIGTESTS (#10), the
builder's core, and the slice files — no collision.

Run:  pytest tests/stubs/test_breaker_stubs.py -v
Expect: RED now; GREEN after #5/#6 land.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: Path, n_bytes: int = 256) -> None:
    """Write a minimal VALID safetensors file: 8-byte little-endian header
    length + that many JSON header bytes (no tensor data needed for a header
    parse). safetensors.safe_open accepts an empty `{}` header."""
    header = b"{}" + b" " * (n_bytes - 2)
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def _write_truncated_safetensors(path: Path) -> None:
    """Write a file that CLAIMS a large header but is cut off mid-header —
    simulates a checkpoint caught mid-write. Header parse must FAIL."""
    claimed = 10_000
    path.write_bytes(struct.pack("<Q", claimed) + b'{"weight":')  # truncated


# ===========================================================================
# F5 — stderr `\r`-terminated tqdm progress is parsed
# ===========================================================================

class TestF5CarriageReturnProgress:
    """Contract (DESIGN §B6): a tqdm bar update arrives on stderr as
    `...\\r 250/2000 ... lr: 1.0e-04 loss: 1.234e-01\\r` with NO trailing `\\n`.
    The reader must split on `[\\r\\n]` and parse_line must extract step/total/loss.
    """

    BAR = " 12%|#2        | 250/2000 [01:23<09:10,  3.18it/s, lr: 1.0e-04 loss: 1.234e-01]"

    def test_step_and_total_parsed_from_bar(self):
        from log_parser import TrainingProgress, parse_line

        prog = TrainingProgress(total_steps=0)
        updated = parse_line(self.BAR, prog)

        assert updated is True, "parse_line must recognize the tqdm bar line"
        assert prog.step == 250, f"expected step=250, got {prog.step}"
        assert prog.total_steps == 2000, f"expected total=2000, got {prog.total_steps}"

    def test_loss_parsed_from_bar(self):
        from log_parser import TrainingProgress, parse_line

        prog = TrainingProgress()
        parse_line(self.BAR, prog)
        assert prog.loss == pytest.approx(0.1234, rel=1e-3), (
            f"expected loss≈0.1234 from 'loss: 1.234e-01', got {prog.loss}"
        )

    def test_reader_splits_on_carriage_return(self):
        """The stderr reader must yield the bar payload even when terminated by
        `\\r` with no `\\n`. A `\\n`-only splitter returns ZERO lines from this
        buffer → progress would stall at 0%. Reconciled to the shipped reader
        `trainer._iter_lines(stream)` (char-by-char, yields on [\\r\\n]).
        """
        import io
        from trainer import _iter_lines

        buf = f"epoch 1/8\n{self.BAR}\r{self.BAR}\r"
        lines = list(_iter_lines(io.StringIO(buf)))
        assert any("250/2000" in ln for ln in lines), (
            "the \\r-terminated bar payload must be yielded as its own line"
        )


# ===========================================================================
# F1 — step-less bare final file is captured and renamed to the final epoch
# ===========================================================================

class TestF1BareFinalFile:
    """Contract (DESIGN §B5/§B4): AI-Toolkit's final save() emits a bare
    `{AITK_JOB_NAME}.safetensors` (wan: `{AITK_JOB_NAME}_high_noise.safetensors`)
    with NO 9-digit step token. The step-pattern regex cannot match it. The
    watcher/final-collect MUST capture it via a bare pattern and map it to
    epoch == total_epochs, then rename via rename_output into `_uploads/`.
    For epochs % save_every_n_epochs != 0 this bare file is the ONLY copy of the
    best-trained weights — dropping it is silent data loss.
    """

    def test_bare_final_matches_a_pattern(self):
        """Reconciled to shipped `trainer._classify(name)` → (kind, step, variant).
        The step regex cannot match the step-less final file; _classify must
        return kind='bare' for it and capture the noise variant."""
        from config import AITK_JOB_NAME
        from trainer import _classify

        kind, step, variant = _classify(f"{AITK_JOB_NAME}.safetensors")
        assert kind == "bare", "step-less final file must classify as 'bare'"
        assert step is None, "bare file has no step token"
        assert variant is None, "non-wan bare file has no noise variant"

        wkind, wstep, wvariant = _classify(f"{AITK_JOB_NAME}_high_noise.safetensors")
        assert wkind == "bare", "wan step-less final must classify as 'bare'"
        assert wvariant == "high", "noise variant must be captured from the bare wan file"

        # and a step file must NOT be misclassified as bare
        skind, sstep, _ = _classify(f"{AITK_JOB_NAME}_000000250.safetensors")
        assert skind == "step" and sstep == 250, "step file classifies as 'step'"

    def test_final_collect_captures_bare_file_into_uploads(self, tmp_path):
        """Drop ONLY the bare final file (the common epochs%save_every!=0 case)
        and assert it is renamed to <trigger>_<branch>_epoch<total_epochs> in
        _uploads/. Reconciled to the shipped seam: _classify('aitk_lora.safetensors')
        -> bare, the watcher maps bare -> epoch=total_epochs (trainer.py:206),
        then _publish copy-renames into _uploads/ via rename_output. We drive
        _publish directly (S3/webhook are no-ops when unconfigured) to prove the
        bare -> final-epoch -> _uploads/ path end to end.
        """
        from config import AITK_JOB_NAME, UPLOADS_SUBDIR, TrainingJob, rename_output
        from log_parser import TrainingProgress
        from trainer import _classify, _publish

        save_dir = tmp_path / "out"
        save_dir.mkdir()
        src = save_dir / f"{AITK_JOB_NAME}.safetensors"
        _write_safetensors(src)

        # the watcher's classification + epoch synthesis for a bare file
        kind, step, variant = _classify(src.name)
        assert kind == "bare"
        total_epochs = 10
        epoch = total_epochs  # trainer.py:206 — bare → total_epochs

        job = TrainingJob(
            job_id="t", model_type="sdxl",
            dataset_zip_url="x", trigger_word="sks",
        )
        uploads = save_dir.parent / UPLOADS_SUBDIR
        _publish(src, epoch, variant, job, TrainingProgress(), uploads)

        expected = rename_output("sks", "sdxl", None, total_epochs)
        assert (uploads / expected).exists(), (
            f"bare final must be copy-renamed to {expected} in {UPLOADS_SUBDIR}/ "
            f"(epoch == total_epochs={total_epochs})"
        )


# ===========================================================================
# N2 — write-completion gate: no upload of a half-written / truncated file
# ===========================================================================

class TestN2WriteCompletionGate:
    """Contract (DESIGN §B5): AI-Toolkit save_file is non-atomic/in-place. The
    watcher must gate each candidate on BOTH (a) size stable across two polls
    AND (b) safetensors header parses, before copy/upload. A growing file or a
    truncated-header file must be SKIPPED (kept pending), never uploaded.

    Reconciled to the shipped trainer API (architect, post-#5):
      - size-stability:  trainer._is_stable(sf, sizes)
      - header-validity: trainer._is_complete_safetensors(path)
    The watcher (_scan_once) requires BOTH before promoting.
    """

    def test_growing_file_not_yet_stable(self, tmp_path):
        from trainer import _is_stable
        p = tmp_path / "aitk_lora_000000250.safetensors"
        _write_safetensors(p, 256)
        sizes: dict = {}
        # first sighting → not stable (no prior size recorded)
        assert _is_stable(p, sizes) is False, "first sighting is never stable"
        # simulate growth before the second poll
        with p.open("ab") as fh:
            fh.write(b"\x00" * 1024)
        assert _is_stable(p, sizes) is False, (
            "a file whose size changed since the last poll is NOT stable"
        )

    def test_stable_then_complete(self, tmp_path):
        from trainer import _is_stable, _is_complete_safetensors
        p = tmp_path / "aitk_lora_000000250.safetensors"
        _write_safetensors(p, 256)
        sizes: dict = {}
        _is_stable(p, sizes)                       # first poll records size
        assert _is_stable(p, sizes) is True, "unchanged size on the 2nd poll is stable"
        assert _is_complete_safetensors(p) is True, (
            "a size-stable, header-valid file passes the completion gate"
        )

    def test_truncated_header_never_complete(self, tmp_path):
        from trainer import _is_complete_safetensors
        p = tmp_path / "aitk_lora_000000250.safetensors"
        _write_truncated_safetensors(p)
        # even if size were "stable", a corrupt/truncated header must fail the gate
        assert _is_complete_safetensors(p) is False, (
            "a truncated/unparseable safetensors header must fail the completion gate"
        )
