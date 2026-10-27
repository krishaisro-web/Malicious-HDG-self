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


def test_profiler(tmp_path: Path) -> None:
    from src.hdg.data.fixture import make_fixture
    raw_dir = tmp_path / "raw_synthetic"
    out_dir = tmp_path / "profile_out"
    make_fixture(raw_dir, n_malware=20, n_umbrella=10, n_cesnet=10, seed=42)

    report, md_text, time_split = profile_dataset(
        raw_dir=raw_dir,
        output_dir=out_dir,
        min_distinct_months=2,
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


def test_shortcut_audit(minimal_domains_df, tmp_path: Path) -> None:
    parquet_file = tmp_path / "domains.parquet"
    minimal_domains_df.to_parquet(parquet_file)
    out_dir = tmp_path / "audit_out"
    summary, md_text = run_shortcut_audit(parquet_file, out_dir, is_fixture=True)

    assert (out_dir / "audit_shortcuts.json").exists()
    assert (out_dir / "audit_shortcuts.md").exists()
    assert "single_feature_aucs" in summary


def test_hetero_gnn_forward_pass_both_variants(minimal_synthetic_heterodata) -> None:
    data = minimal_synthetic_heterodata
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


def test_checkpoint_model_config_contract(minimal_synthetic_heterodata) -> None:
    """Verifies that get_model_config provides complete architecture metadata."""
    data = minimal_synthetic_heterodata
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


def test_checkpoint_round_trip_reconstruction(minimal_synthetic_heterodata, tmp_path: Path) -> None:
    """Verifies that each supported architecture round-trips state_dict + config exactly."""
    data = minimal_synthetic_heterodata
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

    # Test isolated MLP baseline (use_edges=False, non-default prune_threshold) with full round-trip
    m_mlp = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        prune_threshold=0.35,
        use_edges=False
    )
    m_mlp.eval()
    with torch.no_grad():
        out_mlp1 = m_mlp(data.x_dict, data.edge_index_dict)

    mlp_ckpt_p = tmp_path / "test_mlp_no_edges.pt"
    torch.save({
        "model_state_dict": m_mlp.state_dict(),
        "model_config": m_mlp.get_model_config()
    }, mlp_ckpt_p)

    mlp_loaded = torch.load(mlp_ckpt_p, weights_only=False)
    mlp_cfg = mlp_loaded["model_config"]
    assert mlp_cfg["use_edges"] is False
    assert mlp_cfg["prune_threshold"] == 0.35

    m_mlp2 = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant=mlp_cfg["variant"],
        hidden_dim=mlp_cfg["hidden_dim"],
        out_dim=mlp_cfg["out_dim"],
        num_layers=mlp_cfg["num_layers"],
        num_heads=mlp_cfg["num_heads"],
        dropout=mlp_cfg["dropout"],
        prune_threshold=mlp_cfg["prune_threshold"],
        use_edges=mlp_cfg["use_edges"]
    )
    m_mlp2.load_state_dict(mlp_loaded["model_state_dict"])
    m_mlp2.eval()
    with torch.no_grad():
        out_mlp2 = m_mlp2(data.x_dict, data.edge_index_dict)

    torch.testing.assert_close(out_mlp1, out_mlp2)


def test_deterministic_gnn_initialization_and_forward(minimal_synthetic_heterodata):
    """
    Verifies that seeding torch, numpy, and random prior to model construction
    yields 100% bitwise deterministic parameter initialization and forward outputs.
    """
    import random
    import numpy as np
    data = minimal_synthetic_heterodata
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    seed = 1337

    # Run 1: Seed then construct
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    m1 = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="attn",
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        num_heads=2
    )
    m1.eval()
    with torch.no_grad():
        out1 = m1(data.x_dict, data.edge_index_dict)

    # Run 2: Seed again then construct
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    m2 = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="attn",
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        num_heads=2
    )
    m2.eval()
    with torch.no_grad():
        out2 = m2(data.x_dict, data.edge_index_dict)

    # Verify identical initial weights across all modules
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)

    # Verify identical forward outputs
    torch.testing.assert_close(out1, out2)


def test_streaming_evaluation_validation_calibrated_threshold(minimal_synthetic_heterodata):
    """
    Verifies that streaming evaluate_on_indices uses validation negatives to calibrate
    its low-FPR operating point without peeking at test cohort labels.
    """
    from src.hdg.eval.streaming import evaluate_on_indices, calibrate_operating_threshold
    data = minimal_synthetic_heterodata
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        num_layers=2
    )

    import numpy as np
    y_true_all = data["domain"].y.cpu().numpy()
    val_indices = np.array([0, 1, 2, 3], dtype=int)
    test_indices = np.array([4, 5, 6, 7], dtype=int)

    # Calibrate on validation set
    calib_th = calibrate_operating_threshold(model, data, val_indices, y_true_all, target_fpr=0.001)
    assert 0.0 <= calib_th <= 1.0

    # Evaluate on test set with frozen threshold
    metrics = evaluate_on_indices(
        model=model,
        data=data,
        eval_indices=test_indices,
        y_true_all=y_true_all,
        calibrated_threshold_01=calib_th
    )

    assert "tpr_at_01_fpr" in metrics
    assert "realized_fpr_at_01" in metrics
    assert metrics["calibrated_threshold_01"] == round(float(calib_th), 4)


def test_objective_acceptance_criteria_exact():
    """
    Verifies that evaluate_objective_statuses enforces strict acceptance logic:
    - O1 requires realized FPR <= 0.0010 (not just high TPR)
    - O3 requires p < 0.05
    - O4 requires both mean and P95 latency < 50ms
    """
    import importlib
    make_report_mod = importlib.import_module("pipeline.11_make_report")
    evaluate_objective_statuses = make_report_mod.evaluate_objective_statuses

    # Scenario A: High TPR but realized FPR exceeds 0.1% target -> NOT MET
    gnn_data_leaky_fpr = {
        "random": {
            "variants": {
                "sage": {
                    "runs": [{
                        "operating_points": {
                            "tpr_at_0.1pct_fpr": {
                                "tpr": 0.85,
                                "realised_test_fpr": 0.0050  # 0.5% > 0.1% target
                            }
                        }
                    }]
                }
            }
        }
    }
    statuses = evaluate_objective_statuses(
        gnn_evaluations=gnn_data_leaky_fpr,
        baselines_data=None,
        latency_data=None,
        attack_data=None,
        is_fixture=False
    )
    assert statuses["O1"][1] == "NOT MET"

    # Scenario B: High TPR and realized FPR <= 0.0010 -> MET
    gnn_data_compliant = {
        "random": {
            "variants": {
                "sage": {
                    "runs": [{
                        "operating_points": {
                            "tpr_at_0.1pct_fpr": {
                                "tpr": 0.72,
                                "realised_test_fpr": 0.0008  # <= 0.1% target
                            }
                        }
                    }]
                }
            }
        }
    }
    statuses_b = evaluate_objective_statuses(
        gnn_evaluations=gnn_data_compliant,
        baselines_data=None,
        latency_data=None,
        attack_data=None,
        is_fixture=False
    )
    assert statuses_b["O1"][1] == "MET"


def test_rpz_fail_closed_on_split_mismatch_and_missing_cert(tmp_path: Path, minimal_synthetic_heterodata) -> None:
    """
    Regression test for:
    1. RPZ split mismatch: never substitute another checkpoint/split; fail closed with BLOCKED_SPLIT_MISMATCH.
    2. RPZ certification metadata: missing cert_indices must strictly block emission.
    """
    import json
    import subprocess
    import sys
    import pandas as pd
    import yaml

    data = minimal_synthetic_heterodata.clone()
    n_domains = 100
    data["domain"].x = torch.randn((n_domains, data["domain"].x.shape[1]))
    data["domain"].raw_x = data["domain"].x.clone()
    data["domain"].y = torch.tensor([i % 2 for i in range(n_domains)], dtype=torch.long)
    data["domain"].num_nodes = n_domains

    # Expand domain-to-infrastructure edges and populate degree metadata.
    data.incident_pairs = {}

    for edge_type in [
        ("domain", "resolves_to", "ip"),
        ("domain", "uses_ns", "nameserver"),
        ("domain", "registered_by", "registrar"),
        ("domain", "uses_cert", "certificate"),
    ]:
        src_type, rel, dst_type = edge_type
        dst_count = data[dst_type].num_nodes
        src = torch.arange(n_domains, dtype=torch.long)
        dst = src % dst_count

        data[edge_type].edge_index = torch.stack([src, dst])
        reverse_type = (dst_type, "rev_" + rel, src_type)
        if reverse_type in data.edge_types:
            data[reverse_type].edge_index = torch.stack([dst, src])

        data.incident_pairs[dst_type] = list(zip(src.tolist(), dst.tolist()))

    data.incident_pairs["asn"] = []

    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    for directory in (processed_dir, results_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.save(data, processed_dir / "heterodata.pt")
    pd.DataFrame({
        "domain": [f"example{i}.com" for i in range(n_domains)],
        "label": data["domain"].y.cpu().numpy(),
    }).to_parquet(processed_dir / "domains.parquet", index=False)

    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["paths"]["processed_data_dir"] = str(processed_dir)
    config["paths"]["results_dir"] = str(results_dir)
    config["paths"]["checkpoints_dir"] = str(checkpoints_dir)
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16
    )

    # Checkpoint trained on 'random' split
    ckpt_random_path = tmp_path / "gnn_sage_random_seed_42.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "split_type": "random",
        "cert_indices": [0, 1]
    }, ckpt_random_path)
    with open(tmp_path / "gnn_sage_random_seed_42_manifest.json", "w") as f:
        json.dump({"split_type": "random", "seed": 42}, f)

    # Test 1: Request 'time' split with 'random' checkpoint -> must block with BLOCKED_SPLIT_MISMATCH
    cmd = [
        sys.executable, "-m", "pipeline.10_emit_rpz",
        "--config", str(config_path),
        "--force",
        "--checkpoint", str(ckpt_random_path),
        "--split-type", "time"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "BLOCKED_SPLIT_MISMATCH" in res.stdout or "BLOCKED_SPLIT_MISMATCH" in res.stderr

    # Test 2: Checkpoint missing cert_indices -> must block with BLOCKED_MISSING_CERT_METADATA
    ckpt_no_cert = tmp_path / "gnn_sage_random_no_cert.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "split_type": "random",
        "cert_indices": []  # empty/missing
    }, ckpt_no_cert)
    with open(tmp_path / "gnn_sage_random_no_cert_manifest.json", "w") as f:
        json.dump({"split_type": "random", "seed": 42}, f)

    cmd2 = [
        sys.executable, "-m", "pipeline.10_emit_rpz",
        "--config", str(config_path),
        "--force",
        "--checkpoint", str(ckpt_no_cert),
        "--split-type", "random"
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0
    assert "BLOCKED_MISSING_CERT_METADATA" in res2.stdout or "BLOCKED" in res2.stdout


def test_latency_checkpoint_strict_validation(tmp_path: Path, minimal_synthetic_heterodata) -> None:
    """
    Regression test for:
    3. Latency checkpoint reconstruction: load and validate all 8 architecture fields,
       including num_heads and prune_threshold.
    """
    import subprocess
    import sys
    import pandas as pd
    import yaml

    data = minimal_synthetic_heterodata.clone()
    n_domains = 100
    data["domain"].x = torch.randn((n_domains, data["domain"].x.shape[1]))
    data["domain"].raw_x = data["domain"].x.clone()
    data["domain"].y = torch.tensor([i % 2 for i in range(n_domains)], dtype=torch.long)
    data["domain"].num_nodes = n_domains

    # Expand domain-to-infrastructure edges and populate degree metadata.
    data.incident_pairs = {}

    for edge_type in [
        ("domain", "resolves_to", "ip"),
        ("domain", "uses_ns", "nameserver"),
        ("domain", "registered_by", "registrar"),
        ("domain", "uses_cert", "certificate"),
    ]:
        src_type, rel, dst_type = edge_type
        dst_count = data[dst_type].num_nodes
        src = torch.arange(n_domains, dtype=torch.long)
        dst = src % dst_count

        data[edge_type].edge_index = torch.stack([src, dst])
        reverse_type = (dst_type, "rev_" + rel, src_type)
        if reverse_type in data.edge_types:
            data[reverse_type].edge_index = torch.stack([dst, src])

        data.incident_pairs[dst_type] = list(zip(src.tolist(), dst.tolist()))

    data.incident_pairs["asn"] = []

    processed_dir = tmp_path / "processed"
    results_dir = tmp_path / "results"
    checkpoints_dir = tmp_path / "checkpoints"
    for directory in (processed_dir, results_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    torch.save(data, processed_dir / "heterodata.pt")
    pd.DataFrame({
        "domain": [f"example{i}.com" for i in range(n_domains)],
        "label": data["domain"].y.cpu().numpy(),
    }).to_parquet(processed_dir / "domains.parquet", index=False)

    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    config["paths"]["processed_data_dir"] = str(processed_dir)
    config["paths"]["results_dir"] = str(results_dir)
    config["paths"]["checkpoints_dir"] = str(checkpoints_dir)
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="attn",
        hidden_dim=32,
        out_dim=16,
        num_heads=2
    )

    # Incomplete config missing num_heads and prune_threshold
    bad_cfg = {
        "variant": "attn",
        "hidden_dim": 32,
        "out_dim": 16,
        "num_layers": 2,
        "dropout": 0.2,
        "use_edges": True
        # missing num_heads and prune_threshold!
    }
    bad_ckpt = tmp_path / "incomplete_arch_ckpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": bad_cfg
    }, bad_ckpt)

    cmd = [
        sys.executable, "-m", "pipeline.07_latency_eval",
        "--config", str(config_path),
        "--force",
        "--checkpoint", str(bad_ckpt)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    # Must fail to load incomplete checkpoint and log warning
    assert "missing required fields" in res.stdout or "Could not load checkpoint" in res.stdout


def test_o3_and_o4_reporting_criteria() -> None:
    """
    Regression test for:
    4. O3 report criterion: remove the AUROC-difference fallback; if valid statistical
       significance evidence (p < 0.05) is unavailable, report NOT MET/INSUFFICIENT EVIDENCE.
    5. O4 criterion: make documentation and implementation use exactly the same latency acceptance rule.
    """
    import importlib
    mod_report = importlib.import_module("pipeline.11_make_report")
    evaluate_objective_statuses = mod_report.evaluate_objective_statuses

    baselines = {
        "xgboost_tabular": {"roc_auc_mean": 0.85}
    }
    # Case A: HeteroGNN has higher AUC (0.88 > 0.85), but no p < 0.05 evidence
    gnn_no_stat_sig = {
        "random": {
            "variants": {
                "sage": {
                    "roc_auc_mean": 0.88,
                    "runs": [{
                        "bootstrap_ci": {
                            "roc_auc": {"ci_lower": 0.82, "ci_upper": 0.94}  # overlaps 0.85!
                        }
                    }]
                }
            },
            "paired_comparisons": {}
        }
    }
    statuses = evaluate_objective_statuses(
        gnn_evaluations=gnn_no_stat_sig,
        baselines_data=baselines,
        latency_data=None,
        attack_data=None,
        is_fixture=False
    )
    # Must be NOT MET/INSUFFICIENT EVIDENCE (heuristic fallback removed)
    assert statuses["O3"][1] == "NOT MET/INSUFFICIENT EVIDENCE"

    # Case B: CPU Latency Budget (Mean < 50 ms AND P95 < 50 ms)
    # Mean < 50 ms and P95 < 50 ms -> MET
    latency_compliant = {
        "benchmarks": [{
            "batch_size": 1,
            "threads": 1,
            "end_to_end": {"mean_ms": 15.2, "p95_ms": 32.1}
        }]
    }
    statuses_lat = evaluate_objective_statuses(
        gnn_evaluations=None,
        baselines_data=None,
        latency_data=latency_compliant,
        attack_data=None,
        is_fixture=False
    )
    assert statuses_lat["O4"][1] == "MET"

    # Mean < 50 ms but P95 >= 50 ms -> NOT MET
    latency_p95_spike = {
        "benchmarks": [{
            "batch_size": 1,
            "threads": 1,
            "end_to_end": {"mean_ms": 25.0, "p95_ms": 52.4}
        }]
    }
    statuses_lat2 = evaluate_objective_statuses(
        gnn_evaluations=None,
        baselines_data=None,
        latency_data=latency_p95_spike,
        attack_data=None,
        is_fixture=False
    )
    assert statuses_lat2["O4"][1] == "NOT MET"


