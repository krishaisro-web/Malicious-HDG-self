"""
End-to-end integration and smoke tests for Malicious-HDG.
Verifies:
- Profiler emits profile_report.json and profile_report.md (<= 200 lines)
- Shortcut audit identifies zero domain shortcuts on synthetic fixture
- Path safeguard strictly forbids fixture runs from writing to production results/
- HeteroGNN forward pass on HeteroData for both SAGE and Attn variants
"""

from pathlib import Path
import pytest
import torch

from models.hetero_gnn import HeteroGNN
from src.hdg.audit import run_shortcut_audit
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.profiler import profile_dataset


def test_fixture_path_write_safeguard() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=True, root_dir=repo_root)

    # Must raise PermissionError when trying to write to production results
    production_results_path = repo_root / "results" / "profile"
    with pytest.raises(PermissionError, match="Refusing to write fixture output to production path"):
        paths.check_write_path(production_results_path)

    # Must succeed for fixture results
    fixture_results_path = repo_root / "results_fixture" / "profile"
    paths.check_write_path(fixture_results_path)


def test_profiler_on_fixture() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=True, root_dir=repo_root)

    out_dir = paths.results_dir / "profile"
    report, md_text, time_split = profile_dataset(
        raw_dir=paths.raw_dir,
        output_dir=out_dir,
        min_distinct_months=cfg.get("profiling", {}).get("min_distinct_months_for_time_split", 4),
        is_fixture=True
    )

    # Both files must exist
    assert (out_dir / "profile_report.json").exists()
    assert (out_dir / "profile_report.md").exists()
    assert (out_dir / "time_split_valid.json").exists()

    # Markdown must be <= 200 lines
    lines = md_text.splitlines()
    assert len(lines) <= 200

    # Must contain required keys
    assert "classes" in report
    assert "malware" in report["classes"]
    assert "benign_umbrella" in report["classes"]
    assert "benign_cesnet" in report["classes"]


def test_shortcut_audit_on_fixture() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=True, root_dir=repo_root)

    parquet_file = paths.processed_dir / "domains.parquet"
    out_dir = paths.results_dir / "profile"
    summary, md_text = run_shortcut_audit(parquet_file, out_dir, is_fixture=True)

    assert (out_dir / "audit_shortcuts.json").exists()
    assert (out_dir / "audit_shortcuts.md").exists()
    assert summary["verdict_passed"] is True
    # All domain features must be < 0.90
    for feat, score in summary["single_feature_aucs"].items():
        if feat != "source_only":
            assert score < 0.90, f"Feature {feat} exceeded 0.90 AUC threshold!"


def test_hetero_gnn_forward_pass_both_variants() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=True, root_dir=repo_root)

    graph_file = paths.processed_dir / "heterodata.pt"
    data = torch.load(graph_file, weights_only=False)

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    # 1. SAGE
    model_sage = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=32,
        out_dim=16,
        variant="sage"
    )
    model_sage.eval()
    with torch.no_grad():
        out_sage = model_sage(data.x_dict, data.edge_index_dict)
    assert out_sage.shape == (data["domain"].num_nodes, 2)

    # 2. Attn
    model_attn = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=32,
        out_dim=16,
        num_heads=2,
        variant="attn"
    )
    model_attn.eval()
    with torch.no_grad():
        out_attn = model_attn(data.x_dict, data.edge_index_dict)
    assert out_attn.shape == (data["domain"].num_nodes, 2)
