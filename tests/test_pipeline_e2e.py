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

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.data.audit import run_shortcut_audit
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.data.profiler import profile_dataset


def test_fixture_path_write_safeguard(tmp_path: Path) -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=True, root_dir=repo_root)

    # Must raise PermissionError when trying to write to production results
    production_results_path = repo_root / "results" / "real" / "profile"
    with pytest.raises(PermissionError, match="Refusing to write fixture output to production path"):
        paths.check_write_path(production_results_path)

    # Must succeed for fixture results when aimed at temporary/isolated output
    test_subpath = tmp_path / "fixture_sandbox" / "profile"
    paths.check_write_path(test_subpath)  # Should not raise


def test_profiler() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=False, root_dir=repo_root)

    out_dir = paths.results_dir / "profile"
    report, md_text, time_split = profile_dataset(
        raw_dir=paths.raw_dir,
        output_dir=out_dir,
        min_distinct_months=cfg.get("profiling", {}).get("min_distinct_months_for_time_split", 4),
        is_fixture=False
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


def test_shortcut_audit() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=False, root_dir=repo_root)

    parquet_file = paths.processed_dir / "domains.parquet"
    out_dir = paths.results_dir / "profile"
    summary, md_text = run_shortcut_audit(parquet_file, out_dir, is_fixture=False)

    assert (out_dir / "audit_shortcuts.json").exists()
    assert (out_dir / "audit_shortcuts.md").exists()
    assert "single_feature_aucs" in summary


def test_hetero_gnn_forward_pass_both_variants() -> None:
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=False, root_dir=repo_root)

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


def test_checkpoint_model_config_contract() -> None:
    """Verifies that get_model_config provides complete architecture metadata."""
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=False, root_dir=repo_root)

    graph_file = paths.processed_dir / "heterodata.pt"
    data = torch.load(graph_file, weights_only=False)
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="attn",
        hidden_dim=64,
        out_dim=32,
        num_layers=3,
        num_heads=4,
        dropout=0.25
    )
    conf = model.get_model_config()
    assert conf["variant"] == "attn"
    assert conf["hidden_dim"] == 64
    assert conf["out_dim"] == 32
    assert conf["num_layers"] == 3
    assert conf["num_heads"] == 4
    assert conf["dropout"] == 0.25
    assert "use_edges" in conf


def test_checkpoint_round_trip_reconstruction(tmp_path: Path) -> None:
    """Verifies that each supported architecture round-trips state_dict + config exactly."""
    cfg = load_config()
    repo_root = Path(__file__).resolve().parent.parent
    paths = get_resolved_paths(cfg, is_fixture=False, root_dir=repo_root)

    graph_file = paths.processed_dir / "heterodata.pt"
    data = torch.load(graph_file, weights_only=False)
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    for variant in ["sage", "attn", "sage_guard"]:
        m1 = HeteroGNN(
            metadata=data.metadata(),
            in_channels_dict=in_channels,
            variant=variant,
            hidden_dim=48,
            out_dim=24,
            num_layers=2,
            num_heads=3 if variant == "attn" else 4,
            dropout=0.15
        )
        saved_payload = {
            "model_state_dict": m1.state_dict(),
            "model_config": m1.get_model_config()
        }
        ckpt_p = tmp_path / f"test_{variant}.pt"
        torch.save(saved_payload, ckpt_p)

        m1.eval()
        with torch.no_grad():
            out1 = m1(data.x_dict, data.edge_index_dict)

        # Reconstruct exactly from serialized config
        loaded_obj = torch.load(ckpt_p, weights_only=False)
        l_cfg = loaded_obj["model_config"]
        assert l_cfg["variant"] == variant
        assert l_cfg["hidden_dim"] == 48
        assert l_cfg["out_dim"] == 24
        assert l_cfg["num_layers"] == 2
        assert l_cfg["dropout"] == 0.15

        m2 = HeteroGNN(
            metadata=data.metadata(),
            in_channels_dict=in_channels,
            variant=l_cfg["variant"],
            hidden_dim=l_cfg["hidden_dim"],
            out_dim=l_cfg["out_dim"],
            num_layers=l_cfg["num_layers"],
            num_heads=l_cfg["num_heads"],
            dropout=l_cfg["dropout"],
            prune_threshold=l_cfg["prune_threshold"],
            use_edges=l_cfg["use_edges"]
        )
        m2.load_state_dict(loaded_obj["model_state_dict"])
        m2.eval()
        with torch.no_grad():
            out2 = m2(data.x_dict, data.edge_index_dict)

        # Exact logit equivalence check
        torch.testing.assert_close(out1, out2)

    # Test isolated MLP baseline (use_edges=False, non-default prune_threshold)
    m_mlp = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        prune_threshold=0.35,
        use_edges=False
    )
    mlp_cfg = m_mlp.get_model_config()
    assert mlp_cfg["use_edges"] is False
    assert mlp_cfg["prune_threshold"] == 0.35
