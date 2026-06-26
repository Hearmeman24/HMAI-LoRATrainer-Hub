"""Shared test fixtures."""

import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_job_dir(tmp_path):
    """Create a temporary directory mimicking /runpod-volume/jobs/{id}/."""
    job_dir = tmp_path / "jobs" / "test-job-123"
    for sub in ["dataset", "configs", "output", "logs"]:
        (job_dir / sub).mkdir(parents=True)
    return job_dir


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """Create a temp dir with 5 dummy .png + .txt files."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    for i in range(5):
        # Create dummy image files (just need to exist with right extension)
        (ds_dir / f"image_{i}.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
        (ds_dir / f"image_{i}.txt").write_text(f"A photo of a person, image {i}")
    return ds_dir


@pytest.fixture
def sample_dataset_dir_mixed(tmp_path):
    """Create a dataset dir with mixed image extensions."""
    ds_dir = tmp_path / "dataset_mixed"
    ds_dir.mkdir()
    extensions = [".png", ".jpg", ".jpeg", ".webp", ".png"]
    for i, ext in enumerate(extensions):
        (ds_dir / f"image_{i}{ext}").write_bytes(b"\x00" * 100)
        (ds_dir / f"image_{i}.txt").write_text(f"caption {i}")
    return ds_dir


@pytest.fixture
def sample_dataset_dir_missing_captions(tmp_path):
    """Create a dataset dir where some images lack captions."""
    ds_dir = tmp_path / "dataset_missing"
    ds_dir.mkdir()
    for i in range(5):
        (ds_dir / f"image_{i}.png").write_bytes(b"\x00" * 100)
    # Only create captions for first 3
    for i in range(3):
        (ds_dir / f"image_{i}.txt").write_text(f"caption {i}")
    return ds_dir


@pytest.fixture
def sample_zip(tmp_path, sample_dataset_dir):
    """Create a real zip file from sample_dataset_dir."""
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sample_dataset_dir.iterdir():
            zf.write(f, f.name)
    return zip_path


@pytest.fixture
def sample_zip_nested(tmp_path, sample_dataset_dir):
    """Create a zip with files nested in a single folder."""
    zip_path = tmp_path / "dataset_nested.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sample_dataset_dir.iterdir():
            zf.write(f, f"my_dataset/{f.name}")
    return zip_path


@pytest.fixture
def sample_zip_macosx(tmp_path, sample_dataset_dir):
    """Create a zip with __MACOSX entries."""
    zip_path = tmp_path / "dataset_macosx.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sample_dataset_dir.iterdir():
            zf.write(f, f.name)
        # Add __MACOSX junk
        zf.writestr("__MACOSX/._image_0.png", b"junk")
        zf.writestr("__MACOSX/._DS_Store", b"junk")
    return zip_path


@pytest.fixture
def mock_env_s3():
    """Set S3 environment variables."""
    env = {
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "S3_BUCKET": "test-bucket",
        "S3_REGION": "us-east-1",
    }
    with patch.dict(os.environ, env):
        yield env


@pytest.fixture
def mock_env_volume(tmp_path):
    """Set VOLUME_ROOT to a temp directory."""
    env = {"VOLUME_ROOT": str(tmp_path)}
    with patch.dict(os.environ, env):
        yield tmp_path


def _make_payload(model_type, **overrides):
    """Helper to create a valid payload dict."""
    payload = {
        "model_type": model_type,
        "dataset_zip_url": "https://example.com/dataset.zip",
        "trigger_word": "testword01",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def valid_payload_wan22():
    # noise_variant is required for wan2.2 (spec §B/noise_variant gating).
    return _make_payload("wan2.2", noise_variant="high")


@pytest.fixture
def valid_payload_wan22_low():
    return _make_payload("wan2.2", noise_variant="low")


@pytest.fixture
def valid_payload_sdxl():
    return _make_payload("sdxl")


@pytest.fixture
def valid_payload_qwen():
    return _make_payload("qwen_image")


@pytest.fixture
def valid_payload_qwen_2512():
    return _make_payload("qwen_image_2512")


@pytest.fixture
def valid_payload_z_image():
    return _make_payload("z_image")


@pytest.fixture
def valid_payload_ideogram4():
    return _make_payload("ideogram4")


@pytest.fixture
def valid_payload_flux_klein():
    return _make_payload("flux_klein_9b")
