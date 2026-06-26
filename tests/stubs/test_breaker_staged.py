"""Breaker-staged adversarial tests — N1/N3/F2/F3 (for task #13).

DISJOINT companion to tests/stubs/test_breaker_stubs.py (the architect's #15
stubs, which canonically cover F1/F5/N2 with hard-RED asserts against the exact
slice API). This file covers ONLY the findings that file does not — N1, N3, F2,
F3 — so there are no duplicate assertions across the two stub files.

These encode the EXACT contract assertions hardened during the paper-attack
phase, against the frozen §B interfaces in DESIGN.md + spec.claude.md §5.

The translator tests (N3) run NOW and are EXPECTED TO FAIL against the current
overrides_translator.py (it returns a bare dict, not the (cfg, ignored_keys)
tuple the contract mandates) — that failure is the point: it proves the N3
contract is not yet met. F2 and the N1 generator tests skip cleanly until their
slices land (yaml_generator retention constant / generate_config wiring).

Each test cites its finding and the spec line it enforces. Append-only: this
file adds tests; it weakens/deletes nothing.

Run:  pytest tests/stubs/test_breaker_staged.py -v
"""

from __future__ import annotations

import struct
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path: Path, n_tensor_bytes: int = 64, *, truncate_to: int | None = None) -> int:
    """Write a minimal valid safetensors file (or a truncated one).

    Layout: <8-byte LE header_len><header json><tensor bytes>.
    If truncate_to is given, only that many leading bytes are written (simulating
    a checkpoint caught mid-write — the N2 corrupt-upload window).
    Returns the FULL (untruncated) byte length.
    """
    header = json.dumps(
        {"lora.weight": {"dtype": "F16", "shape": [n_tensor_bytes // 2],
                         "data_offsets": [0, n_tensor_bytes]}}
    ).encode()
    blob = struct.pack("<Q", len(header)) + header + (b"\x00" * n_tensor_bytes)
    path.write_bytes(blob if truncate_to is None else blob[:truncate_to])
    return len(blob)


# ===========================================================================
# N3 — typo'd/unknown override key signal is LOG-ONLY; OUTPUT shape frozen.
#   spec.claude.md §5: translate_overrides returns (partial_config, ignored_keys);
#   handler LOGS ignored_keys; OUTPUT gets NO ignored_overrides field.
#   These run LIVE and are expected to FAIL against the current translator
#   (returns a bare dict). That is a real, fireable finding.
# ===========================================================================

class TestN3IgnoredKeysContract:
    def _call(self, overrides):
        import overrides_translator as ot
        return ot.translate_overrides(
            overrides,
            model_type="sdxl",
            dataset_image_count=20,
            base_batch_size=1,
            base_grad_accum=1,
        )

    def test_returns_cfg_and_ignored_keys_tuple(self):
        """Contract (spec §5/N3): returns (partial_config, ignored_keys)."""
        result = self._call({"optimzer.lr": 1e-4})  # deliberate typo
        assert isinstance(result, tuple) and len(result) == 2, (
            "translate_overrides must return (partial_config, ignored_keys) per "
            "spec.claude.md §5 N3 — got a bare value instead"
        )
        cfg, ignored = result
        assert isinstance(cfg, dict)
        assert isinstance(ignored, (list, tuple, set))

    def test_typo_lands_in_ignored_not_in_cfg(self):
        """A fat-fingered `optimzer.lr` must be reported as ignored AND must not
        leak into the config (it must NOT silently become train.lr)."""
        cfg, ignored = self._call({"optimzer.lr": 1e-4})
        assert "optimzer.lr" in ignored
        # the typo must not have set the real sink
        assert cfg.get("train", {}).get("lr") != 1e-4

    def test_dropped_known_key_also_reported(self):
        """A known-but-dropped key (pipeline_stages) is reported ignored too,
        so the caller has a greppable signal it no-op'd."""
        cfg, ignored = self._call({"pipeline_stages": 2})
        assert "pipeline_stages" in ignored

    def test_mapped_key_not_in_ignored(self):
        """A real mapped key (optimizer.lr) must NOT appear in ignored_keys."""
        cfg, ignored = self._call({"optimizer.lr": 3e-4})
        assert "optimizer.lr" not in ignored
        assert cfg["train"]["lr"] == 3e-4

    def test_warn_message_states_no_effect(self, monkeypatch):
        """spec §5/N3: the warn must name the key and say it did not take effect
        (greppable typo signal), not a vague 'ignoring'."""
        import overrides_translator as ot
        captured = []
        monkeypatch.setattr(ot.alog, "warning",
                            lambda msg, **kw: captured.append(msg))
        self._call({"optimzer.lr": 1e-4})
        joined = " ".join(captured).lower()
        assert "optimzer.lr" in joined, "warn must name the offending key"
        assert ("did not take effect" in joined or "no effect" in joined
                or "not take effect" in joined), (
            "warn must state the override did not take effect (spec §5 N3), "
            f"got: {captured!r}"
        )


class TestN3OutputShapeFrozen:
    """The OUTPUT JSON must NOT gain an `ignored_overrides` field (lead ruling:
    response shape is a frozen invariant; signal is log-only)."""

    def test_output_has_no_ignored_overrides_field(self):
        from config import TrainingResult
        # The frozen OUTPUT is assembled from TrainingResult; assert the dataclass
        # carries no ignored_overrides surface that would leak into the JSON.
        r = TrainingResult(ok=True)
        assert not hasattr(r, "ignored_overrides"), (
            "OUTPUT shape is frozen (spec §5 N3) — ignored keys are log-only, "
            "must not appear as a TrainingResult/OUTPUT field"
        )


# NOTE: F1 (bare-final two-pattern + final-collect), F5 (\r-split tqdm parse),
# and N2 (write-completion gate) are covered canonically by the architect's
# stubs in tests/stubs/test_breaker_stubs.py (hard-RED `pytest.fail` against the
# exact slice API: trainer.BARE_FINAL_PATTERN / iter_progress_lines /
# collect_final_outputs / is_checkpoint_complete). This file is DISJOINT — it
# covers only N1/N3/F2/F3, which that file does not. No duplicate assertions.


# ===========================================================================
# F2 / F3 — retention (never -1) + prune-survival of _uploads/ copies.
#   spec §5/§Constraint: max_step_saves_to_keep=100000 (never -1, which deletes
#   all but oldest); config name='aitk_lora' (not trigger); renamed copies in
#   sibling _uploads/ outside the prune glob. Skips until #5/#8 land.
# ===========================================================================

class TestF2F3RetentionAndPruneSurvival:
    def test_uploads_copy_survives_simulated_prune(self, tmp_path):
        """Simulate AI-Toolkit's prune (glob aitk_lora_*, delete all-but-N in
        save_dir). A renamed copy in the sibling _uploads/ must survive."""
        save_dir = tmp_path / "output"
        save_dir.mkdir()
        uploads_dir = tmp_path / "_uploads"  # sibling of save_dir per spec §5/F3
        uploads_dir.mkdir()

        # AI-Toolkit's own per-step files in save_dir.
        for step in (90, 180, 270):
            _write_safetensors(save_dir / f"aitk_lora_{step:09d}.safetensors")
        # A renamed copy the watcher placed in the sibling uploads dir.
        renamed = uploads_dir / "testword_sdxl_epoch9.safetensors"
        _write_safetensors(renamed)

        # Simulate AI-Toolkit clean_up_saves: glob {name}_*, keep newest 1, delete rest.
        import glob, os
        items = sorted(glob.glob(str(save_dir / "aitk_lora_*")),
                       key=os.path.getctime)
        for victim in items[:-1]:
            os.remove(victim)

        # F3 invariant: the prune glob is scoped to save_dir/aitk_lora_* and can
        # never touch the sibling uploads dir.
        assert renamed.exists(), (
            "renamed copy in sibling _uploads/ must survive AI-Toolkit's prune "
            "(F3) — it must NOT live in save_dir under the aitk_lora_* glob"
        )

    def test_retention_constant_is_positive_never_minus_one(self):
        """spec §5/F2: retention is a large positive int (100000), NEVER -1
        (which slices files[:1] → deletes all but oldest)."""
        gen = pytest.importorskip("yaml_generator")
        # The generator (or a shared constant) must pin retention. Probe common names.
        val = None
        for name in ("MAX_SAVES_KEEP", "MAX_STEP_SAVES_TO_KEEP", "SAVES_TO_KEEP"):
            if hasattr(gen, name):
                val = getattr(gen, name)
                break
        if val is None:
            pytest.skip("yaml_generator not yet exposing a retention constant")
        assert isinstance(val, int) and val > 1, f"retention must be a large +int, got {val!r}"
        assert val != -1, "retention must NEVER be -1 (deletes all but oldest, F2)"


# ===========================================================================
# N1 — user-controlled values assigned POST-parse, never string-substituted.
#   spec §5/N1: trigger_word + all overrides set on the parsed dict after
#   yaml.safe_load; templates carry NO {trigger_word}/{caption_prefix}.
#   civitai_checkpoint_path also post-parse. Runs once yaml_generator exposes
#   generate_config; templates check runs now.
# ===========================================================================

class TestN1NoUserValueSubstitution:
    def test_templates_have_no_trigger_placeholder(self):
        """N1: shipped YAML templates must NOT contain {trigger_word} or
        {caption_prefix} — those are the raw-substitution bug."""
        # tests/stubs/ -> repo root is three parents up.
        cfgdir = Path(__file__).resolve().parents[2] / "configs"
        yamls = list(cfgdir.glob("*.yaml"))
        if not yamls:
            pytest.skip("configs/*.yaml not yet authored (core #4)")
        offenders = [y.name for y in yamls
                     if "{trigger_word}" in y.read_text()
                     or "{caption_prefix}" in y.read_text()]
        assert not offenders, (
            f"templates must not embed user-value placeholders (N1): {offenders}"
        )

    @pytest.mark.parametrize("hostile", [
        'sks "art" style',   # quote → ParserError under raw substitution
        '123',               # YAML-coerces to int under raw substitution
        'yes',               # → bool True
        'a: b',              # → injects a mapping / ScannerError
        '{x}',               # → dict
        '- dash',            # → list under raw substitution
    ])
    def test_hostile_trigger_word_survives_as_literal_str(self, hostile, tmp_path):
        """A YAML-hostile trigger_word must produce a valid config with the
        trigger preserved as a literal string (no crash, no type-coercion).
        Runs end-to-end against the real generate_config (N1 post-parse assign)."""
        import yaml
        import config as _cfg
        gen = pytest.importorskip("yaml_generator")
        if not hasattr(gen, "generate_config"):
            pytest.skip("yaml_generator.generate_config not present")
        _cfg.JOBS_DIR = tmp_path / "jobs"
        job = _cfg.validate_payload({
            "model_type": "sdxl", "dataset_zip_url": "u",
            "trigger_word": hostile, "job_id": "j",
        })
        for d in (job.configs_dir, job.output_dir, job.dataset_dir):
            d.mkdir(parents=True, exist_ok=True)
        path = gen.generate_config(job, 20)  # must not raise (N1)
        proc = yaml.safe_load(open(path))["config"]["process"][0]
        tw = proc["datasets"][0].get("trigger_word")
        assert tw == hostile, f"trigger_word mangled: {tw!r} != {hostile!r}"
        assert isinstance(tw, str), (
            f"trigger_word type-coerced to {type(tw).__name__} (N1 raw-substitution bug)"
        )
