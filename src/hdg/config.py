"""
Configuration loader, path validation, and run management for Malicious-HDG.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import yaml


@dataclass
class ResolvedPaths:
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    is_fixture: bool
    is_smoke: bool = False

    def check_write_path(self, target_path: Path) -> None:
        """Enforces that fixture runs never write to real results."""
        resolved = target_path.resolve()
        if self.is_fixture and "results_fixture" not in str(resolved) and "data_fixture" not in str(resolved):
            raise PermissionError(
                f"[FIXTURE SAFEGUARD] Refusing to write fixture output to production path: {resolved}. "
                f"Fixture output must only go to results_fixture/ or data_fixture/."
            )


def init_thread_pool(cfg: Optional[Dict[str, Any]] = None) -> int:
    """Sets PyTorch thread count from HDG_THREADS, config, or cpu_count(). Returns configured thread count."""
    env_threads = os.environ.get("HDG_THREADS")
    if env_threads is not None:
        num_threads = int(env_threads)
    elif cfg is not None and "threads" in cfg:
        num_threads = int(cfg["threads"])
    elif cfg is not None and "hardware" in cfg and "num_threads" in cfg["hardware"]:
        num_threads = int(cfg["hardware"]["num_threads"])
    else:
        num_threads = os.cpu_count() or 1
    torch.set_num_threads(num_threads)
    return num_threads


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if config_path is None:
        default_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        config_path = str(default_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_resolved_paths(
    cfg: Dict[str, Any],
    is_fixture: bool = False,
    is_smoke: bool = False,
    root_dir: Optional[Path] = None
) -> ResolvedPaths:
    """
    Returns resolved filesystem paths depending on whether fixture and/or smoke mode is enabled.
    Guarantees strict isolation between test/fixture/smoke environments and production data.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    p = cfg["paths"]
    if is_fixture:
        raw_dir = root_dir / p["fixture_raw_dir"]
        processed_dir = root_dir / p["fixture_processed_dir"]
        base_results = root_dir / p["fixture_results_dir"]
    else:
        raw_dir = root_dir / p["raw_data_dir"]
        processed_dir = root_dir / p["processed_data_dir"]
        base_results = root_dir / p["results_dir"]

    results_dir = (base_results / "smoke") if is_smoke else base_results

    return ResolvedPaths(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        results_dir=results_dir,
        is_fixture=is_fixture,
        is_smoke=is_smoke
    )


def _resolve_results_dir(target: Union[Path, ResolvedPaths]) -> Path:
    if isinstance(target, ResolvedPaths):
        return target.results_dir
    return target


def get_run_output_path(results_dir: Union[Path, ResolvedPaths], run_name: str) -> Path:
    """Returns path for an individual experiment run JSON."""
    r_dir = _resolve_results_dir(results_dir)
    runs_dir = r_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"{run_name}.json"


def should_skip_run(results_dir: Union[Path, ResolvedPaths], run_name: str, force: bool = False) -> bool:
    """Checks if run JSON already exists to allow resumability unless --force is specified."""
    if force:
        return False
    path = get_run_output_path(results_dir, run_name)
    return path.exists() and path.stat().st_size > 0


def save_run_result(results_dir: Union[Path, ResolvedPaths], run_name: str, data: Dict[str, Any]) -> Path:
    """Persists an individual run JSON for resumability and report aggregation."""
    path = get_run_output_path(results_dir, run_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
