"""Model download management for AI-Toolkit models.

Resolves each ModelSpec.downloads list (repo / url / hf_file items) into
MODELS_DIR, then sets/resolves name_or_path so the YAML generator receives
a ready local path or HF repo id.

Download kinds (DownloadItem.kind):
  "repo"    — full HuggingFace repo to MODELS_DIR/<local_subdir>
  "url"     — direct URL (CivitAI, HF resolve, etc.) to MODELS_DIR/<local_subdir>/<filename>
  "hf_file" — single file from a HF repo: <repo_id>/resolve/main/<filename>

Download strategy (mirrors comfyui-wan / comfyui-qwen-image):
  - HuggingFace URLs/repos go through huggingface_hub (hf_hub_download /
    snapshot_download) with hf_xet acceleration. This is resumable,
    integrity-checked, and avoids the per-chunk 403 noise aria2c hits against
    HF's Xet CDN.
  - Non-HF URLs (CivitAI, arbitrary direct links) fall back to aria2c with
    multi-connection downloads.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger

import config as cfg
from config import DownloadItem, ModelSpec, TrainingJob

# aria2c tuning (non-HF downloads only) — 16 connections per server
ARIA2_CONNECTIONS = "16"
ARIA2_SPLIT = "16"


# ---------------------------------------------------------------------------
# aria2c helpers (non-HF downloads only)
# ---------------------------------------------------------------------------

def ensure_aria2c() -> None:
    """Install aria2c if not already available."""
    if subprocess.run(["which", "aria2c"], capture_output=True).returncode == 0:
        return
    logger.info("Installing aria2c...")
    subprocess.run(["apt-get", "update", "-qq"], capture_output=True)
    result = subprocess.run(
        ["apt-get", "install", "-y", "-qq", "aria2"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install aria2: {result.stderr}")
    logger.info("aria2c installed")


def _has_model_files(model_dir: Path) -> bool:
    """Return True if model_dir has at least one substantial (>1 KB) file."""
    if not model_dir.exists():
        return False
    return any(f.is_file() and f.stat().st_size > 1000 for f in model_dir.rglob("*"))


# ---------------------------------------------------------------------------
# HuggingFace helpers (hf_hub_download / snapshot_download + hf_xet)
# ---------------------------------------------------------------------------

def _is_hf_url(url: str) -> bool:
    return urlparse(url).netloc == "huggingface.co"


def _parse_hf_url(url: str) -> tuple[str, str, str]:
    """Split `.../{owner}/{repo}/resolve/{revision}/{path}` -> (repo_id, revision, filename)."""
    parts = urlparse(url).path.lstrip("/").split("/")
    if len(parts) < 5 or parts[2] != "resolve":
        raise ValueError(f"Unrecognized HF URL shape: {url}")
    repo_id = f"{parts[0]}/{parts[1]}"
    revision = parts[3]
    filename = "/".join(parts[4:])
    return repo_id, revision, filename


def _hf_download_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
    out_name: str | None = None,
    revision: str = "main",
) -> Path:
    """Download a single file from a HF repo via huggingface_hub (hf_xet accelerated)."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"hf_hub_download {repo_id}/{filename} @ {revision} -> {local_dir}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
    )
    out = local_dir / (out_name or Path(filename).name)
    if Path(downloaded).resolve() != out.resolve():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), str(out))
    return out


# ---------------------------------------------------------------------------
# Per-kind download primitives
# ---------------------------------------------------------------------------

def _download_repo(repo_id: str, local_dir: Path) -> None:
    """Download a full HuggingFace repo to local_dir via snapshot_download.

    Uses huggingface_hub (hf_xet accelerated, resumable, integrity-checked);
    already-cached files are skipped by the hub layer.
    """
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {repo_id} to {local_dir} (snapshot_download)")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), token=token)
    logger.info(f"Downloaded {repo_id}")


def _download_url(url: str, local_dir: Path, local_filename: str) -> None:
    """Download a single file.

    HF resolve URLs route through huggingface_hub; everything else (CivitAI,
    arbitrary direct links) uses aria2c with multi-connection downloads.
    """
    target_path = local_dir / local_filename

    if target_path.exists() and target_path.stat().st_size > 1000:
        logger.info(f"File already exists: {target_path}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace -> huggingface_hub (no aria2c Xet-CDN 403 noise)
    if _is_hf_url(url):
        repo_id, revision, filename = _parse_hf_url(url)
        _hf_download_file(repo_id, filename, local_dir, local_filename, revision)
        logger.info(f"Downloaded {local_filename}")
        return

    # Non-HF -> aria2c
    logger.info(f"Downloading {url} (aria2c)")
    ensure_aria2c()
    cmd = [
        "aria2c",
        f"--max-connection-per-server={ARIA2_CONNECTIONS}",
        f"--split={ARIA2_SPLIT}",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=warn",
        "--retry-wait=2",
        "--max-tries=5",
        f"--dir={local_dir}",
        f"--out={local_filename}",
        url,
    ]
    logger.info(f"aria2c downloading {local_filename} ({ARIA2_CONNECTIONS} connections)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} (aria2c exit {result.returncode})")

    logger.info(f"Downloaded {local_filename}")


def _download_hf_file(
    repo_id: str, filename: str, local_dir: Path, local_filename: str | None = None
) -> None:
    """Download a single file from a HuggingFace repo via huggingface_hub."""
    _hf_download_file(repo_id, filename, local_dir, local_filename or Path(filename).name)


# ---------------------------------------------------------------------------
# Primary entry points
# ---------------------------------------------------------------------------

def _resolve_download_item(item: DownloadItem) -> Path:
    """Execute one DownloadItem and return the local path it landed at."""
    local_subdir = item.local_subdir or ""
    local_dir = cfg.MODELS_DIR / local_subdir if local_subdir else cfg.MODELS_DIR

    if item.kind == "repo":
        if not item.repo_id:
            raise ValueError(f"DownloadItem kind='repo' missing repo_id: {item}")
        # Skip full download if the directory already has model files.
        if _has_model_files(local_dir):
            logger.info(f"Repo {item.repo_id} already present at {local_dir}")
        else:
            _download_repo(item.repo_id, local_dir)
        return local_dir

    elif item.kind == "url":
        if not item.url or not item.filename:
            raise ValueError(f"DownloadItem kind='url' requires url + filename: {item}")
        _download_url(url=item.url, local_dir=local_dir, local_filename=item.filename)
        return local_dir / item.filename

    elif item.kind == "hf_file":
        if not item.repo_id or not item.filename:
            raise ValueError(f"DownloadItem kind='hf_file' requires repo_id + filename: {item}")
        _download_hf_file(
            repo_id=item.repo_id,
            filename=item.filename,
            local_dir=local_dir,
            local_filename=item.filename,
        )
        return local_dir / item.filename

    else:
        raise ValueError(f"Unknown DownloadItem kind: {item.kind!r}")


def _maybe_download_civitai(job: TrainingJob, spec: ModelSpec) -> Path | None:
    """Download a CivitAI checkpoint if job.civitai_model_id is set.

    Returns the local .safetensors path, or None if no civitai download needed.
    N1 constraint: civitai_checkpoint_path is set post-safe_load, never
    string-substituted (SOURCE_FINDINGS §8).
    """
    if not job.civitai_model_id:
        return None

    civitai_api_key = os.environ.get("CIVITAI_API_KEY", "")
    civitai_dir = cfg.MODELS_DIR / "civitai"
    civitai_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the model version download URL from the CivitAI API.
    api_url = f"https://civitai.com/api/v1/models/{job.civitai_model_id}"
    headers = {"User-Agent": "hmai-loratrainer-hub/1.0"}
    if civitai_api_key:
        headers["Authorization"] = f"Bearer {civitai_api_key}"
    req = urllib.request.Request(api_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch CivitAI model {job.civitai_model_id}: {e}") from e

    # Take the first file from the latest model version.
    model_versions = data.get("modelVersions", [])
    if not model_versions:
        raise RuntimeError(f"No versions found for CivitAI model {job.civitai_model_id}")

    version_files = model_versions[0].get("files", [])
    if not version_files:
        raise RuntimeError(
            f"No files in latest version of CivitAI model {job.civitai_model_id}"
        )

    download_url = version_files[0]["downloadUrl"]
    if civitai_api_key:
        download_url = f"{download_url}?token={civitai_api_key}"

    local_filename = version_files[0].get("name", f"civitai_{job.civitai_model_id}.safetensors")
    local_path = civitai_dir / local_filename

    if local_path.exists() and local_path.stat().st_size > 1000:
        logger.info(f"CivitAI model already present: {local_path}")
    else:
        _download_url(url=download_url, local_dir=civitai_dir, local_filename=local_filename)

    return local_path


def ensure_model(job: TrainingJob) -> str:
    """Ensure all model assets for job are present locally.

    Resolves each DownloadItem in job.model_spec.downloads, handles CivitAI
    for sdxl, and returns the resolved name_or_path string the YAML generator
    should use for model.name_or_path.

    For diffusers repos: returns the local directory path string.
    For sdxl single-file (CivitAI or direct URL): returns the local .safetensors path.
    For sdxl base (no CivitAI): returns the HF repo id (AI-Toolkit will from_pretrained).
    """
    spec = job.model_spec

    # Handle CivitAI for sdxl first — overrides the normal download if present.
    if job.civitai_model_id:
        civitai_path = _maybe_download_civitai(job, spec)
        if civitai_path is not None:
            # Store on job for use by yaml_generator (N1: assigned post-parse, not
            # string-substituted).
            job.civitai_checkpoint_path = str(civitai_path)
            logger.info(f"CivitAI path: {civitai_path}")
            # No need to download the base sdxl model; from_single_file dispatches
            # automatically on a file path (SOURCE_FINDINGS §8).
            return str(civitai_path)

    # Resolve each DownloadItem in order.
    resolved_paths: list[Path] = []
    for item in spec.downloads:
        path = _resolve_download_item(item)
        resolved_paths.append(path)

    # name_or_path resolution: the spec carries the canonical answer (repo id for
    # diffusers models that AI-Toolkit loads via from_pretrained; the local path
    # is the fallback for single-file/adapter cases).
    if spec.name_or_path:
        return spec.name_or_path

    # Fallback: use the first resolved local path.
    return str(resolved_paths[0]) if resolved_paths else ""
