"""
Configuration loader and path validation for Malicious-HDG.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class ResolvedPaths:
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    is_fixture: bool

    def check_write_path(self, target_path: Path) -> None:
        """Enforces that fixture runs never write to real results."""
        resolved = target_path.resolve()
        if self.is_fixture and "results_fixture" not in str(resolved) and "data_fixture" not in str(resolved):
            raise PermissionError(
                f"[FIXTURE SAFEGUARD] Refusing to write fixture output to production path: {resolved}. "
                f"Fixture output must only go to results_fixture/ or data_fixture/."
            )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if config_path is None:
        default_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        config_path = str(default_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_resolved_paths(cfg: Dict[str, Any], is_fixture: bool = False, root_dir: Optional[Path] = None) -> ResolvedPaths:
    """
    Returns resolved filesystem paths depending on whether fixture mode is enabled.
    Guarantees strict isolation between test/fixture environments and production data.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    p = cfg["paths"]
    if is_fixture:
        raw_dir = root_dir / p["fixture_raw_dir"]
        processed_dir = root_dir / p["fixture_processed_dir"]
        results_dir = root_dir / p["fixture_results_dir"]
    else:
        raw_dir = root_dir / p["raw_data_dir"]
        processed_dir = root_dir / p["processed_data_dir"]
        results_dir = root_dir / p["results_dir"]

    return ResolvedPaths(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        results_dir=results_dir,
        is_fixture=is_fixture
    )
