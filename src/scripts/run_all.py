#!/usr/bin/env python3
"""
End-to-end pipeline runner for Malicious-HDG.
Executes the full pipeline sequence:
1. Dataset profiling (00_profile_dataset)
2. Normalization & parsing (01_parse)
3. Shortcut audit (00b_audit_shortcuts)
4. Heterogeneous graph build (02_build_graph)
5. Baselines evaluation (03_run_baselines)
6. HeteroGNN multi-split evaluation (04_train_gnn)
7. Ablation study (05_ablation)
8. Structural adversarial attack (06_attack_eval)
9. CPU latency benchmark (07_latency_eval)

Supports --fixture for fast CPU validation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.fixture import make_fixture


def run_cmd(cmd: List[str]) -> None:
    print(f"\n[RUN] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Malicious-HDG pipeline.")
    parser.add_argument("--fixture", action="store_true", help="Execute against data_fixture/ (smoke test)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for GNN models")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"{pfx}Starting complete Malicious-HDG pipeline...")

    # Step 0: Ensure fixture exists if running in fixture mode
    if args.fixture:
        mal_f = paths.raw_dir / "malware.json"
        if not mal_f.exists():
            print(f"[FIXTURE] Generating initial schema fixture at {paths.raw_dir}...")
            make_fixture(paths.raw_dir, n_malware=1500, n_umbrella=750, n_cesnet=750, seed=42)

    py_exe = sys.executable

    # 1. Profile Dataset
    profile_cmd = [py_exe, "-m", "src.scripts.00_profile_dataset"]
    if args.fixture:
        profile_cmd.append("--fixture")
    run_cmd(profile_cmd)

    # 2. Parse & Normalize
    parse_cmd = [py_exe, "-m", "src.scripts.01_parse"]
    if args.fixture:
        parse_cmd.append("--fixture")
    run_cmd(parse_cmd)

    # 3. Shortcut Audit
    audit_cmd = [py_exe, "-m", "src.scripts.00b_audit_shortcuts"]
    if args.fixture:
        audit_cmd.append("--fixture")
    run_cmd(audit_cmd)

    # 4. Build Graph
    graph_cmd = [py_exe, "-m", "src.scripts.02_build_graph"]
    if args.fixture:
        graph_cmd.append("--fixture")
    run_cmd(graph_cmd)

    # 5. Baselines
    base_cmd = [py_exe, "-m", "src.scripts.03_run_baselines"]
    if args.fixture:
        base_cmd.append("--fixture")
    if args.single_seed is not None:
        base_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture:
        base_cmd.extend(["--single-seed", "42"])
    run_cmd(base_cmd)

    # 6. GNN Evaluation
    epochs_val = args.epochs if args.epochs is not None else (3 if args.fixture else 30)
    gnn_cmd = [
        py_exe, "-m", "src.scripts.04_train_gnn",
        "--variant", "both",
        "--epochs", str(epochs_val)
    ]
    if args.fixture:
        gnn_cmd.append("--fixture")
    if args.single_seed is not None:
        gnn_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture:
        gnn_cmd.extend(["--single-seed", "42"])
    run_cmd(gnn_cmd)

    # 7. Ablation Study
    abl_cmd = [
        py_exe, "-m", "src.scripts.05_ablation",
        "--epochs", str(epochs_val)
    ]
    if args.fixture:
        abl_cmd.append("--fixture")
    if args.single_seed is not None:
        abl_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture:
        abl_cmd.extend(["--single-seed", "42"])
    run_cmd(abl_cmd)

    # 8. Structural Attack
    atk_cmd = [
        py_exe, "-m", "src.scripts.06_attack_eval",
        "--epochs", str(epochs_val)
    ]
    if args.fixture:
        atk_cmd.append("--fixture")
    run_cmd(atk_cmd)

    # 9. Latency Benchmark
    lat_cmd = [
        py_exe, "-m", "src.scripts.07_latency_eval",
        "--num-queries", "20" if args.fixture else "100"
    ]
    if args.fixture:
        lat_cmd.append("--fixture")
    run_cmd(lat_cmd)

    print(f"\n{pfx}==================================================")
    print(f"{pfx}Malicious-HDG complete pipeline finished successfully!")
    print(f"{pfx}All artifacts generated in: {paths.results_dir}")
    print(f"{pfx}==================================================")


if __name__ == "__main__":
    main()
