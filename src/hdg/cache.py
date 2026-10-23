"""
Content-aware caching, experiment provenance fingerprinting, and atomic serialization for Malicious-HDG.
Provides:
- compute_file_sha256: Fast buffered SHA-256 calculation.
- compute_dict_fingerprint: Canonical deterministic SHA-256 for configuration/metadata dicts.
- compute_experiment_fingerprint: Combined cryptographic fingerprint across code, config, inputs, seeds.
- is_run_cache_valid: Verifies whether an existing run JSON matches current experiment fingerprint.
- atomic_save_torch: Atomically saves PyTorch objects using NamedTemporaryFile + fsync + os.replace.
- atomic_save_json: Atomically saves JSON manifests to prevent partial-write corruption.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def get_git_info() -> Dict[str, Any]:
    """Retrieves current Git commit and working-tree dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_out) > 0
    except Exception:
        is_dirty = False

    return {"commit": commit, "is_dirty": is_dirty}


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hex digest of a file in chunks."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot compute fingerprint: file does not exist: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_fingerprint(data: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hex digest of a JSON-serializable dictionary."""
    canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


def compute_experiment_fingerprint(
    *args,
    stage: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    input_files: Optional[List[Union[str, Path]]] = None,
    split_type: Optional[str] = None,
    model_variant: Optional[str] = None,
    seed: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_dict: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Computes a cryptographic experiment fingerprint encompassing configuration,
    input artifact checksums, split protocol, architecture, and seeds.
    Supports flexible positional and keyword ordering (e.g. stage first or config first).
    """
    for arg in args:
        if isinstance(arg, str) and stage is None:
            stage = arg
        elif isinstance(arg, dict) and config is None:
            config = arg

    stage = stage or kwargs.get("stage_name") or "unknown_stage"
    config = config or {}
    extra = extra_params or extra_dict or {}

    # Root Cause J: Research configuration fingerprint is invariant to runtime hardware settings
    # (threads, detected RAM, paths, device), but sensitive to scientific research parameters.
    research_config = {
        k: v for k, v in config.items()
        if k not in ("hardware", "runtime", "paths", "system", "device", "num_threads", "num_workers")
    }

    git_info = get_git_info()
    elements: Dict[str, Any] = {
        "stage": stage,
        "git_commit": git_info.get("commit", "unknown")[:10],
        "config_hash": compute_dict_fingerprint(research_config),
        "split_type": split_type or "none",
        "model_variant": model_variant or "none",
        "seed": seed if seed is not None else "none",
        "extra": extra
    }

    if input_files:
        file_hashes = {}
        for f in input_files:
            fp = Path(f)
            if fp.exists():
                file_hashes[fp.name] = compute_file_sha256(fp)[:16]
            else:
                file_hashes[fp.name] = "missing"
        elements["input_hashes"] = file_hashes

    return compute_dict_fingerprint(elements)



def is_run_cache_valid(
    run_file: Union[str, Path],
    expected_fingerprint: str,
    force: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Validates whether a cached run JSON artifact matches the expected experiment fingerprint.
    Returns (is_valid, reason).
    """
    if force:
        return False, "Forced recomputation (--force flag specified)"

    p = Path(run_file)
    if not p.exists() or p.stat().st_size == 0:
        return False, "Cached run file does not exist or is empty"

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to parse cached JSON: {e}"

    cached_status = data.get("status")
    if cached_status in ("FAILED",):
        return False, f"Cached run recorded a FAILED status: {data.get('error')}"

    cached_fp = data.get("fingerprint") or data.get("metadata", {}).get("experiment_fingerprint")
    if cached_fp is None:
        # Legacy run without fingerprint
        return False, "Cached run artifact lacks content-aware experiment fingerprint"

    if cached_fp != expected_fingerprint:
        return False, f"Fingerprint mismatch: cached={cached_fp}, current={expected_fingerprint}"

    return True, f"Cache valid (fingerprint: {cached_fp})"


def atomic_save_torch(state: Any, target_path: Union[str, Path]) -> Path:
    """
    Atomically saves a PyTorch object using a temporary file in the same directory,
    fsyncs OS-level buffers, and renames with os.replace to prevent partial-write corruption.
    """
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".tmp") as tf:
        temp_path = Path(tf.name)
        try:
            torch.save(state, tf)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            tf.close()
            if temp_path.exists():
                temp_path.unlink()
            raise

    os.replace(temp_path, target)
    return target


def atomic_save_json(data: Dict[str, Any], target_path: Union[str, Path], indent: int = 2) -> Path:
    """Atomically writes JSON payload to target_path using os.replace."""
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False, suffix=".tmp", encoding="utf-8") as tf:
        temp_path = Path(tf.name)
        try:
            json.dump(data, tf, indent=indent, default=str)
            tf.flush()
            os.fsync(tf.fileno())
        except Exception:
            tf.close()
            if temp_path.exists():
                temp_path.unlink()
            raise

    os.replace(temp_path, target)
    return target
