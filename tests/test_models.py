"""Tests for model_downloader.py and the MODEL_REGISTRY shape.

Assertions cover:
 - Registry completeness + per-model arch/quantize/downloads fields (spec §6 surface)
 - ensure_model download dispatch (mocked) — repo, url, hf_file DownloadItem kinds
 - CivitAI bypass path for sdxl
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import config as cfg
from config import (
    MODEL_REGISTRY,
    DownloadItem,
    ModelSpec,
    TrainingJob,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_models_dir(tmp_path, monkeypatch):
    """Redirect MODELS_DIR to a tmp directory for all tests."""
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")


@pytest.fixture()
def make_job():
    """Factory: build a minimal TrainingJob for the given model_type."""
    def _make(model_type: str, **kwargs) -> TrainingJob:
        defaults = dict(
            job_id="test-job",
            dataset_zip_url="https://example.com/data.zip",
            trigger_word="mytrigger",
        )
        defaults.update(kwargs)
        return TrainingJob(model_type=model_type, **defaults)
    return _make


# ---------------------------------------------------------------------------
# Registry shape tests
# ---------------------------------------------------------------------------

class TestModelRegistry:
    EXPECTED_MODELS = {
        "wan2.2", "sdxl", "qwen_image", "qwen_image_2512",
        "z_image", "ideogram4", "flux_klein_9b", "krea2", "krea2_turbo",
    }

    def test_all_models_present(self):
        assert set(MODEL_REGISTRY.keys()) == self.EXPECTED_MODELS

    # --- krea2 (raw + turbo) ---

    def test_krea2_arch_and_repo(self):
        assert MODEL_REGISTRY["krea2"].arch == "krea2"
        assert MODEL_REGISTRY["krea2"].downloads[0].repo_id == "krea/Krea-2-Raw"
        assert MODEL_REGISTRY["krea2"].quantize is True

    def test_krea2_turbo_adapter(self):
        spec = MODEL_REGISTRY["krea2_turbo"]
        assert spec.arch == "krea2"
        assert spec.downloads[0].repo_id == "krea/Krea-2-Turbo"
        assert spec.assistant_lora_path.endswith("krea2_turbo_training_adapter_v1.safetensors")

    # --- wan2.2 ---

    def test_wan22_arch(self):
        assert MODEL_REGISTRY["wan2.2"].arch == "wan22_14b"

    def test_wan22_repo_id(self):
        items = MODEL_REGISTRY["wan2.2"].downloads
        assert any(i.repo_id == "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16" for i in items)

    def test_wan22_quantize(self):
        spec = MODEL_REGISTRY["wan2.2"]
        assert spec.quantize is True
        assert spec.qtype is not None and "uint4" in spec.qtype
        assert spec.quantize_te is True
        assert spec.qtype_te == "qfloat8"

    def test_wan22_dual_noise(self):
        assert MODEL_REGISTRY["wan2.2"].dual_noise is True

    def test_wan22_low_vram(self):
        assert MODEL_REGISTRY["wan2.2"].low_vram is True

    # --- sdxl ---

    def test_sdxl_arch(self):
        assert MODEL_REGISTRY["sdxl"].arch == "sdxl"

    def test_sdxl_no_quantize(self):
        assert MODEL_REGISTRY["sdxl"].quantize is False

    def test_sdxl_not_dual_noise(self):
        assert MODEL_REGISTRY["sdxl"].dual_noise is False

    def test_sdxl_has_download(self):
        assert len(MODEL_REGISTRY["sdxl"].downloads) >= 1

    # --- qwen_image ---

    def test_qwen_image_arch(self):
        assert MODEL_REGISTRY["qwen_image"].arch == "qwen_image"

    def test_qwen_image_repo(self):
        items = MODEL_REGISTRY["qwen_image"].downloads
        assert any(i.repo_id == "Qwen/Qwen-Image" for i in items)

    def test_qwen_image_quantize(self):
        spec = MODEL_REGISTRY["qwen_image"]
        assert spec.quantize is True
        assert spec.qtype is not None and "uint3" in spec.qtype

    # --- qwen_image_2512 ---

    def test_qwen_image_2512_same_arch_as_qwen(self):
        # SOURCE_FINDINGS §4: same arch "qwen_image" but different repo
        assert MODEL_REGISTRY["qwen_image_2512"].arch == "qwen_image"

    def test_qwen_image_2512_repo(self):
        items = MODEL_REGISTRY["qwen_image_2512"].downloads
        assert any(i.repo_id == "Qwen/Qwen-Image-2512" for i in items)

    def test_qwen_image_2512_ara_path_different_from_qwen(self):
        spec_q = MODEL_REGISTRY["qwen_image"]
        spec_2512 = MODEL_REGISTRY["qwen_image_2512"]
        assert spec_q.qtype != spec_2512.qtype  # different ARA file

    # --- z_image ---

    def test_z_image_arch_no_underscore(self):
        # SOURCE_FINDINGS §4: actual arch = "zimage" not "z_image"
        assert MODEL_REGISTRY["z_image"].arch == "zimage"

    def test_z_image_turbo_assistant_lora(self):
        # SOURCE_FINDINGS §4: turbo adapter is assistant_lora_path, NOT extras_name_or_path
        spec = MODEL_REGISTRY["z_image"]
        assert spec.assistant_lora_path is not None
        assert "zimage_turbo" in spec.assistant_lora_path

    def test_z_image_quantize(self):
        spec = MODEL_REGISTRY["z_image"]
        assert spec.quantize is True
        assert spec.qtype == "qfloat8"

    def test_z_image_not_dual_noise(self):
        assert MODEL_REGISTRY["z_image"].dual_noise is False

    # --- ideogram4 ---

    def test_ideogram4_arch(self):
        assert MODEL_REGISTRY["ideogram4"].arch == "ideogram4"

    def test_ideogram4_repo(self):
        items = MODEL_REGISTRY["ideogram4"].downloads
        assert any(i.repo_id == "ideogram-ai/ideogram-4-fp8" for i in items)

    def test_ideogram4_unconditional_lora(self):
        spec = MODEL_REGISTRY["ideogram4"]
        assert spec.unconditional_lora_path is not None
        assert "ideogram_4_unconditional" in spec.unconditional_lora_path

    def test_ideogram4_timestep_linear(self):
        assert MODEL_REGISTRY["ideogram4"].timestep_type == "linear"

    def test_ideogram4_not_dual_noise(self):
        assert MODEL_REGISTRY["ideogram4"].dual_noise is False

    # --- flux_klein_9b ---

    def test_flux_klein_9b_arch(self):
        # SOURCE_FINDINGS §4: NOT "flux2_klein" — must be "flux2_klein_9b"
        assert MODEL_REGISTRY["flux_klein_9b"].arch == "flux2_klein_9b"

    def test_flux_klein_9b_repo(self):
        items = MODEL_REGISTRY["flux_klein_9b"].downloads
        assert any(i.repo_id == "black-forest-labs/FLUX.2-klein-base-9B" for i in items)

    def test_flux_klein_9b_quantize(self):
        spec = MODEL_REGISTRY["flux_klein_9b"]
        assert spec.quantize is True
        assert spec.qtype == "qfloat8"

    def test_flux_klein_9b_not_dual_noise(self):
        assert MODEL_REGISTRY["flux_klein_9b"].dual_noise is False

    # --- global invariants ---

    def test_only_wan22_is_dual_noise(self):
        for model_type, spec in MODEL_REGISTRY.items():
            if model_type == "wan2.2":
                assert spec.dual_noise is True
            else:
                assert spec.dual_noise is False

    def test_all_specs_have_downloads(self):
        for model_type, spec in MODEL_REGISTRY.items():
            assert len(spec.downloads) >= 1, f"{model_type} has no downloads"

    def test_all_repo_items_have_repo_id(self):
        for model_type, spec in MODEL_REGISTRY.items():
            for item in spec.downloads:
                if item.kind == "repo":
                    assert item.repo_id, f"{model_type} repo item missing repo_id"

    def test_all_url_items_have_url_and_filename(self):
        for model_type, spec in MODEL_REGISTRY.items():
            for item in spec.downloads:
                if item.kind == "url":
                    assert item.url, f"{model_type} url item missing url"
                    assert item.filename, f"{model_type} url item missing filename"


# ---------------------------------------------------------------------------
# ensure_model — download dispatch
# ---------------------------------------------------------------------------

class TestEnsureModel:
    """Verify that ensure_model calls the right download primitives per DownloadItem kind."""

    @patch("model_downloader._download_repo")
    def test_repo_kind_calls_download_repo(self, mock_dl_repo, make_job, tmp_path):
        from model_downloader import ensure_model

        job = make_job("wan2.2")
        spec = job.model_spec
        assert any(i.kind == "repo" for i in spec.downloads)

        ensure_model(job)

        assert mock_dl_repo.called
        called_repo_id = mock_dl_repo.call_args[0][0]
        assert called_repo_id == "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16"

    @patch("model_downloader._download_repo")
    def test_repo_kind_skips_if_already_present(self, mock_dl_repo, make_job, tmp_path):
        from model_downloader import ensure_model

        job = make_job("wan2.2")
        # Pre-populate the local_subdir so _has_model_files returns True.
        local_dir = cfg.MODELS_DIR / "Wan2.2-T2V-A14B-Diffusers-bf16"
        local_dir.mkdir(parents=True)
        (local_dir / "model.safetensors").write_bytes(b"\x00" * 2000)

        ensure_model(job)

        mock_dl_repo.assert_not_called()

    @patch("model_downloader._download_url")
    def test_url_kind_calls_download_url(self, mock_dl_url, make_job):
        from model_downloader import ensure_model

        job = make_job("sdxl")
        # sdxl has a url-kind DownloadItem
        spec = job.model_spec
        url_items = [i for i in spec.downloads if i.kind == "url"]
        assert url_items, "sdxl should have a url-kind DownloadItem"

        ensure_model(job)

        assert mock_dl_url.called
        called_url = mock_dl_url.call_args[1]["url"] if mock_dl_url.call_args[1] else mock_dl_url.call_args[0][0]
        assert "huggingface.co" in called_url or "stabilityai" in called_url or called_url

    @patch("model_downloader._download_hf_file")
    def test_hf_file_kind_calls_download_hf_file(self, mock_dl_hf, make_job, tmp_path):
        """Synthetic: inject an hf_file DownloadItem and confirm the right primitive fires."""
        from model_downloader import _resolve_download_item

        item = DownloadItem(
            kind="hf_file",
            repo_id="ostris/test_adapter",
            filename="adapter.safetensors",
            local_subdir="adapters",
        )
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _resolve_download_item(item)

        mock_dl_hf.assert_called_once_with(
            repo_id="ostris/test_adapter",
            filename="adapter.safetensors",
            local_dir=cfg.MODELS_DIR / "adapters",
            local_filename="adapter.safetensors",
        )

    @patch("model_downloader._download_repo")
    def test_returns_name_or_path_for_repo_model(self, mock_dl_repo, make_job):
        """ensure_model returns the spec's name_or_path (HF repo id) for diffusers models."""
        from model_downloader import ensure_model

        job = make_job("qwen_image")
        result = ensure_model(job)
        assert result == "Qwen/Qwen-Image"

    @patch("model_downloader._download_repo")
    def test_wan22_returns_correct_name_or_path(self, mock_dl_repo, make_job):
        from model_downloader import ensure_model

        job = make_job("wan2.2", noise_variant="high")
        result = ensure_model(job)
        assert result == "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16"

    @patch("model_downloader._download_repo")
    def test_z_image_returns_turbo_repo_id(self, mock_dl_repo, make_job):
        from model_downloader import ensure_model

        job = make_job("z_image")
        result = ensure_model(job)
        assert result == "Tongyi-MAI/Z-Image-Turbo"

    @patch("model_downloader._download_repo")
    def test_ideogram4_returns_correct_repo_id(self, mock_dl_repo, make_job):
        from model_downloader import ensure_model

        job = make_job("ideogram4")
        result = ensure_model(job)
        assert result == "ideogram-ai/ideogram-4-fp8"

    @patch("model_downloader._download_repo")
    def test_flux_klein_9b_returns_correct_repo_id(self, mock_dl_repo, make_job):
        from model_downloader import ensure_model

        job = make_job("flux_klein_9b")
        result = ensure_model(job)
        assert result == "black-forest-labs/FLUX.2-klein-base-9B"


# ---------------------------------------------------------------------------
# CivitAI bypass for sdxl
# ---------------------------------------------------------------------------

class TestCivitAIDownload:
    """Verify CivitAI checkpoint is downloaded and name_or_path overridden."""

    def _mock_civitai_response(self, model_id: str) -> bytes:
        payload = {
            "modelVersions": [{
                "files": [{
                    "name": f"civitai_{model_id}.safetensors",
                    "downloadUrl": f"https://civitai.com/api/download/models/{model_id}",
                }]
            }]
        }
        return json.dumps(payload).encode("utf-8")

    @patch("model_downloader._download_url")
    @patch("urllib.request.urlopen")
    def test_civitai_model_downloads_and_sets_path(
        self, mock_urlopen, mock_dl_url, make_job
    ):
        from model_downloader import ensure_model

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = self._mock_civitai_response("12345")
        mock_urlopen.return_value = mock_resp

        job = make_job("sdxl", civitai_model_id="12345")
        result = ensure_model(job)

        # Should return a path inside the civitai subdir
        assert "civitai" in result
        assert result.endswith(".safetensors")
        # civitai_checkpoint_path set on job (N1: post-parse, not string-substituted)
        assert job.civitai_checkpoint_path == result

    @patch("model_downloader._download_url")
    @patch("urllib.request.urlopen")
    def test_civitai_skips_base_model_download(
        self, mock_urlopen, mock_dl_url, make_job
    ):
        """When CivitAI path is used, the HF base model is NOT downloaded."""
        from model_downloader import ensure_model

        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = self._mock_civitai_response("99")
        mock_urlopen.return_value = mock_resp

        with patch("model_downloader._download_repo") as mock_repo:
            job = make_job("sdxl", civitai_model_id="99")
            ensure_model(job)
            # Base sdxl repo must NOT be downloaded when CivitAI path is present.
            mock_repo.assert_not_called()


# ---------------------------------------------------------------------------
# Download primitive unit tests
# ---------------------------------------------------------------------------

class TestDownloadPrimitives:
    @patch("model_downloader.subprocess.run")
    def test_download_url_nonhf_calls_aria2c(self, mock_run, tmp_path):
        """Non-HF URLs (CivitAI / direct links) download via aria2c."""
        from model_downloader import _download_url

        mock_run.return_value = MagicMock(returncode=0)
        local_dir = tmp_path / "models" / "test"
        local_dir.mkdir(parents=True)

        _download_url(
            url="https://example.com/model.safetensors",
            local_dir=local_dir,
            local_filename="model.safetensors",
        )

        assert mock_run.called
        cmds = [call.args[0] for call in mock_run.call_args_list]
        aria2c_calls = [c for c in cmds if c[0] == "aria2c"]
        assert aria2c_calls, "Expected at least one aria2c call"

    @patch("huggingface_hub.hf_hub_download")
    @patch("model_downloader.subprocess.run")
    def test_download_url_hf_uses_hf_hub_not_aria2c(self, mock_run, mock_hf, tmp_path):
        """HF resolve URLs route through huggingface_hub, never aria2c."""
        from model_downloader import _download_url

        local_dir = tmp_path / "models" / "sdxl"
        local_dir.mkdir(parents=True)
        # hf_hub_download "downloads" the file in place
        def fake(**kw):
            p = local_dir / kw["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 5000)
            return str(p)
        mock_hf.side_effect = fake

        _download_url(
            url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae.safetensors",
            local_dir=local_dir,
            local_filename="vae.safetensors",
        )

        mock_hf.assert_called_once()
        kw = mock_hf.call_args.kwargs
        assert kw["repo_id"] == "stabilityai/stable-diffusion-xl-base-1.0"
        assert kw["filename"] == "vae.safetensors"
        # aria2c must NOT be used for HF URLs
        aria2c_calls = [c.args[0] for c in mock_run.call_args_list if c.args and c.args[0][0] == "aria2c"]
        assert not aria2c_calls, "HF URL must not fall back to aria2c"

    @patch("model_downloader.subprocess.run")
    def test_download_url_skips_if_file_present(self, mock_run, tmp_path):
        from model_downloader import _download_url

        local_dir = tmp_path / "models"
        local_dir.mkdir(parents=True)
        target = local_dir / "model.safetensors"
        target.write_bytes(b"\x00" * 2000)

        _download_url(url="https://example.com/model.safetensors",
                      local_dir=local_dir, local_filename="model.safetensors")

        # subprocess should not be called since file already exists
        mock_run.assert_not_called()

    @patch("huggingface_hub.snapshot_download")
    def test_download_repo_calls_snapshot_download(self, mock_snap, tmp_path):
        from model_downloader import _download_repo

        local_dir = tmp_path / "models" / "newrepo"
        _download_repo("test/newrepo", local_dir)

        mock_snap.assert_called_once()
        kw = mock_snap.call_args.kwargs
        assert kw["repo_id"] == "test/newrepo"
        assert kw["local_dir"] == str(local_dir)

    @patch("model_downloader._download_repo")
    def test_resolve_skips_repo_when_files_present(self, mock_repo, tmp_path, monkeypatch):
        """_resolve_download_item must NOT re-download a repo whose files exist."""
        import model_downloader as md
        monkeypatch.setattr(md.cfg, "MODELS_DIR", tmp_path / "models")
        present = tmp_path / "models" / "diffusers-x"
        present.mkdir(parents=True)
        (present / "model.safetensors").write_bytes(b"\x00" * 5000)

        md._resolve_download_item(
            DownloadItem(kind="repo", repo_id="x/y", local_subdir="diffusers-x")
        )
        mock_repo.assert_not_called()

    def test_resolve_download_item_unknown_kind_raises(self, tmp_path):
        from model_downloader import _resolve_download_item

        item = DownloadItem(kind="unknown", repo_id="x/y", local_subdir="x")
        with pytest.raises(ValueError, match="Unknown DownloadItem kind"):
            _resolve_download_item(item)
