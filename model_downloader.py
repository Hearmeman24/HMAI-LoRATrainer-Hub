"""Model download management for AI-Toolkit models.

Resolves each ModelSpec.downloads list (repo / url / hf_file items) into
MODELS_DIR, then sets/resolves name_or_path so the YAML generator receives
a ready local path or HF repo id.

Download kinds (DownloadItem.kind):
  "repo"    — full HuggingFace repo cloned to MODELS_DIR/<local_subdir>
  "url"     — direct URL (CivitAI, HF resolve, etc.) to MODELS_DIR/<local_subdir>/<filename>
  "hf_file" — single file from a HF repo: <repo_id>/resolve/main/<filename>

aria2c is used for all downloads — 8 connections per server, 4 files in
parallel for repo bulk downloads; 8 connections for single-file downloads.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

import config as cfg
from config import DownloadItem, ModelSpec, TrainingJob

# aria2c tuning — 8 connections per server, 4 files in parallel
ARIA2_CONNECTIONS = "8"
ARIA2_SPLIT = "8"
ARIA2_PARALLEL_FILES = "4"
HF_API_BASE = "https://huggingface.co/api"
HF_CDN_BASE = "https://huggingface.co"


# ---------------------------------------------------------------------------
# aria2c helpers
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


def _hf_auth_header() -> list[str]:
    """aria2c --header arg list for HF Bearer auth; empty if no token."""
    token = os.environ.get("HF_TOKEN")
    return [f"--header=Authorization: Bearer {token}"] if token else []


def _list_repo_files(repo_id: str, revision: str = "main") -> list[dict]:
    """List all files in a HuggingFace repo via the HF tree API.

    Returns a list of dicts with at least 'path' and 'size' for each file
    (directories excluded). Raises RuntimeError on auth or network failure.
    """
    url = f"{HF_API_BASE}/models/{repo_id}/tree/{revision}?recursive=true"
    headers = {"User-Agent": "hmai-loratrainer-hub/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            entries = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                f"Auth error listing {repo_id}. Set HF_TOKEN for gated repos."
            ) from e
        raise RuntimeError(f"HTTP {e.code} listing {repo_id}: {e.reason}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to list {repo_id}: {e}") from e

    return [e for e in entries if e.get("type") == "file"]


def _aria2c_batch(input_file: Path, log_label: str) -> None:
    """Run aria2c against a batch-input file."""
    ensure_aria2c()
    cmd = [
        "aria2c",
        f"--max-connection-per-server={ARIA2_CONNECTIONS}",
        f"--split={ARIA2_SPLIT}",
        f"--max-concurrent-downloads={ARIA2_PARALLEL_FILES}",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=warn",
        "--summary-interval=5",
        "--retry-wait=2",
        "--max-tries=5",
        "--input-file", str(input_file),
    ]
    logger.info(
        f"aria2c batch {log_label}: split={ARIA2_SPLIT} "
        f"connections={ARIA2_CONNECTIONS} parallel_files={ARIA2_PARALLEL_FILES}"
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"aria2c batch failed for {log_label} (exit {result.returncode})"
        )


# ---------------------------------------------------------------------------
# Per-kind download primitives
# ---------------------------------------------------------------------------

def _download_repo(repo_id: str, local_dir: Path) -> None:
    """Download a full HuggingFace repo to local_dir using aria2c.

    Resolves the file list via the HF tree API, then batch-downloads with 8
    connections per server and 4 files in parallel, preserving repo paths.
    Already-present files (size matches) are skipped.
    """
    logger.info(f"Downloading {repo_id} to {local_dir} (aria2c)")
    local_dir.mkdir(parents=True, exist_ok=True)

    files = _list_repo_files(repo_id)
    if not files:
        raise RuntimeError(f"No files found in repo {repo_id}")

    token = os.environ.get("HF_TOKEN")
    auth_lines: list[str] = []
    if token:
        auth_lines = [f"  header=Authorization: Bearer {token}"]

    # Build aria2c input file: one URL + per-file options block per entry.
    pending: list[str] = []
    for entry in files:
        rel_path = entry["path"]
        size = entry.get("size", 0)
        target = local_dir / rel_path
        if target.exists() and target.stat().st_size == size and size > 0:
            logger.debug(f"  skip (already present): {rel_path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_CDN_BASE}/{repo_id}/resolve/main/{rel_path}"
        pending.append(url)
        pending.append(f"  dir={target.parent}")
        pending.append(f"  out={target.name}")
        pending.extend(auth_lines)

    if not pending:
        logger.info(f"Repo {repo_id} already fully present")
        return

    input_file = local_dir / ".aria2c_input.txt"
    input_file.write_text("\n".join(pending) + "\n")
    try:
        _aria2c_batch(input_file, f"repo={repo_id}")
    finally:
        input_file.unlink(missing_ok=True)

    logger.info(f"Downloaded {repo_id}")


def _download_url(url: str, local_dir: Path, local_filename: str) -> None:
    """Download a file from a direct URL using aria2c (8 connections)."""
    target_path = local_dir / local_filename

    if target_path.exists() and target_path.stat().st_size > 1000:
        logger.info(f"File already exists: {target_path}")
        return

    logger.info(f"Downloading {url}")
    local_dir.mkdir(parents=True, exist_ok=True)

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
    ]

    if "huggingface.co" in url:
        cmd.extend(_hf_auth_header())

    cmd.append(url)

    logger.info(f"aria2c downloading {local_filename} ({ARIA2_CONNECTIONS} connections)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        target_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} (aria2c exit {result.returncode})")

    logger.info(f"Downloaded {local_filename}")


def _download_hf_file(repo_id: str, filename: str, local_dir: Path, local_filename: str | None = None) -> None:
    """Download a single file from a HuggingFace repo via aria2c."""
    target_name = local_filename or Path(filename).name
    url = f"{HF_CDN_BASE}/{repo_id}/resolve/main/{filename}"
    _download_url(url=url, local_dir=local_dir, local_filename=target_name)


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

    # name_or_path resolution:
    # - diffusers repos → use local directory (first repo item that landed)
    # - url single-file (sdxl base) → use the HF repo id since AI-Toolkit will
    #   from_pretrained from the repo; the local file we downloaded is its
    #   single-file variant but the spec's name_or_path is the canonical answer.
    # The YAML generator reads spec.name_or_path directly; for repo-kind items
    # the spec already carries the correct repo id. The local path is the fallback
    # for turbo/assistant adapter paths (those are HF paths auto-downloaded by aitk).
    if spec.name_or_path:
        return spec.name_or_path

    # Fallback: use the first resolved local path.
    return str(resolved_paths[0]) if resolved_paths else ""
