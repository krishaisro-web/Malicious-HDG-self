"""
Unit and regression tests for content-aware caching, atomic persistence,
and profile isolation (Root Causes E & F).
"""

from pathlib import Path
import json
import pytest
import torch

from src.hdg.config import (
    load_config,
    get_resolved_paths,
    should_skip_run,
    save_run_result,
    ResolvedPaths
)
from src.hdg.cache import (
    compute_dict_fingerprint,
    compute_experiment_fingerprint,
    atomic_save_json,
    atomic_save_torch
)


def test_research_configuration_fingerprint_invariance_to_hardware():
    """
    Root Cause J: Research fingerprint must be invariant to runtime hardware settings
    (CPU thread allocation, detected RAM), but sensitive to research parameters (model dim, seeds).
    """
    base_cfg = {
        "randomness": {"seed": 42, "evaluation_seeds": [42, 1337]},
        "model": {"sage": {"hidden_dim": 64, "out_dim": 32}},
        "graph": {"max_domains": 50000},
        "hardware": {"num_threads": 4, "detected_ram_gb": 32.0}
    }

    # Different hardware settings (e.g. running on ISRO HP Thin Client with 2 threads, 8GB RAM)
    thin_client_cfg = {
        "randomness": {"seed": 42, "evaluation_seeds": [42, 1337]},
        "model": {"sage": {"hidden_dim": 64, "out_dim": 32}},
        "graph": {"max_domains": 50000},
        "hardware": {"num_threads": 2, "detected_ram_gb": 8.0}
    }

    fp_base = compute_experiment_fingerprint(base_cfg, stage="train_gnn")
    fp_thin = compute_experiment_fingerprint(thin_client_cfg, stage="train_gnn")

    # Research fingerprint MUST match because the research experiment is identical
    assert fp_base == fp_thin

    # Changing model hidden_dim MUST change the research fingerprint
    modified_model_cfg = {
        "randomness": {"seed": 42, "evaluation_seeds": [42, 1337]},
        "model": {"sage": {"hidden_dim": 32, "out_dim": 32}},
        "graph": {"max_domains": 50000},
        "hardware": {"num_threads": 4, "detected_ram_gb": 32.0}
    }
    fp_mod = compute_experiment_fingerprint(modified_model_cfg, stage="train_gnn")
    assert fp_mod != fp_base


def test_content_aware_run_skipping(tmp_path: Path):
    """
    Root Cause F: A run is only reused when the experiment fingerprint matches.
    Stale cache records are rejected.
    """
    run_name = "test_experiment"
    exp_fp = "sha256_abcdef123456"

    # Initially does not exist -> do not skip
    assert not should_skip_run(tmp_path, run_name, expected_fingerprint=exp_fp)

    # Save run result with matching fingerprint
    save_run_result(tmp_path, run_name, {"experiment_fingerprint": exp_fp, "accuracy": 0.95})

    # Now matching -> should skip
    assert should_skip_run(tmp_path, run_name, expected_fingerprint=exp_fp)

    # If --force is passed -> do not skip
    assert not should_skip_run(tmp_path, run_name, force=True, expected_fingerprint=exp_fp)

    # If expected fingerprint changed -> do NOT skip (stale!)
    assert not should_skip_run(tmp_path, run_name, expected_fingerprint="sha256_new_fingerprint")


def test_atomic_file_saving(tmp_path: Path):
    """Verifies that atomic_save_json and atomic_save_torch leave no temporary corrupt files."""
    json_path = tmp_path / "test.json"
    data_dict = {"status": "SUCCESS", "score": 42}
    atomic_save_json(data_dict, json_path)

    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data_dict

    # PyTorch atomic save
    torch_path = tmp_path / "tensor.pt"
    tensor = torch.tensor([1, 2, 3, 4])
    atomic_save_torch(tensor, torch_path)

    assert torch_path.exists()
    loaded_t = torch.load(torch_path, weights_only=True)
    assert torch.equal(loaded_t, tensor)


def test_profile_path_isolation_and_safeguards(tmp_path: Path):
    """
    Root Cause E: Strict isolation between Real, Smoke, and Fixture profiles.
    check_write_path MUST raise PermissionError if smoke/fixture attempts to write to production.
    """
    cfg = {
        "paths": {
            "raw_data_dir": "data/raw/zenodo",
            "processed_data_dir": "data/processed/real",
            "results_dir": "results/real",
            "checkpoints_dir": "artifacts/checkpoints/real",
            "smoke_processed_dir": "data/processed/smoke",
            "smoke_results_dir": "results/smoke",
            "smoke_checkpoints_dir": "artifacts/checkpoints/smoke",
            "fixture_raw_dir": "data/fixture/zenodo",
            "fixture_processed_dir": "data/processed/fixture",
            "fixture_results_dir": "results/fixture",
            "fixture_checkpoints_dir": "artifacts/checkpoints/fixture",
        }
    }

    # 1. Smoke profile routes to smoke paths
    smoke_paths = get_resolved_paths(cfg, is_smoke=True, root_dir=tmp_path)
    assert "smoke" in str(smoke_paths.processed_dir)
    assert "smoke" in str(smoke_paths.results_dir)
    assert "smoke" in str(smoke_paths.checkpoints_dir)

    # 2. Fixture profile routes to fixture paths
    fixture_paths = get_resolved_paths(cfg, is_fixture=True, root_dir=tmp_path)
    assert "fixture" in str(fixture_paths.processed_dir)
    assert "fixture" in str(fixture_paths.results_dir)
    assert "fixture" in str(fixture_paths.checkpoints_dir)

    # 3. Safeguard: Smoke run cannot write to production real path!
    with pytest.raises(PermissionError, match="\\[SMOKE SAFEGUARD\\]"):
        smoke_paths.check_write_path(tmp_path / "results/real/output.json")

    # 4. Safeguard: Fixture run cannot write to production real path!
    with pytest.raises(PermissionError, match="\\[FIXTURE SAFEGUARD\\]"):
        fixture_paths.check_write_path(tmp_path / "results/real/output.json")
