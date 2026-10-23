#!/usr/bin/env python3
"""
End-to-end pipeline runner for Malicious-HDG.
Executes the full experimental sequence:
0. Preflight checks & resource validation (00c_preflight)
1. Dataset profiling (00_profile_dataset)
2. Normalization & parsing (01_parse)
3. Shortcut audit (00b_audit_shortcuts)
4. Heterogeneous base graph build (02_build_graph)
5. Baselines evaluation (03_run_baselines)
6. HeteroGNN multi-split evaluation (04_train_gnn)
7. Ablation study (05_ablation)
8. Structural adversarial attack & GNNGuard (06_attack_eval)
9. CPU latency benchmark (07_latency_eval)
10. Provenance checks & shortcut audits (08_provenance_checks)
11. Streaming continual learning evaluation (09_streaming_eval)
12. BIND RPZ rule emission (10_emit_rpz)
13. Comprehensive report compilation (11_make_report)

Supports --fixture for fast CPU validation and --smoke for accelerated tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.data.fixture import make_fixture


import time

def run_step(step_num: int, total_steps: int, title: str, cmd: List[str]) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"▶ STEP [{step_num:02d}/{total_steps:02d}]: {title.upper()}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{bar}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"\n❌ STEP [{step_num:02d}/{total_steps:02d}] FAILED after {elapsed:.2f}s with exit code {res.returncode}")
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
    print(f"✔ STEP [{step_num:02d}/{total_steps:02d}] COMPLETED in {elapsed:.2f}s\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete Malicious-HDG pipeline.")
    parser.add_argument("--fixture", action="store_true", help="Execute against data_fixture/ (smoke test)")
    parser.add_argument("--smoke", action="store_true", help="Run in accelerated smoke mode")
    parser.add_argument("--thin-client", action="store_true", help="Run with HP Thin Client profile (configs/hp_thin_client.yaml)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for GNN models")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed only")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domain nodes")
    parser.add_argument("--force", action="store_true", help="Force rerun of all completed stages")
    args = parser.parse_args()

    if args.thin_client and args.config is None:
        args.config = str(REPO_ROOT / "configs" / "hp_thin_client.yaml")

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"{pfx}Starting complete Malicious-HDG pipeline...")

    # Step 0: Ensure fixture exists if running in fixture mode
    if args.fixture:
        mal_f = paths.raw_dir / "malware.json"
        if not mal_f.exists():
            print(f"[FIXTURE] Generating initial schema fixture at {paths.raw_dir}...")
            make_fixture(paths.raw_dir, n_malware=1500, n_umbrella=750, n_cesnet=750, seed=42)

    py_exe = sys.executable

    common_flags: List[str] = []
    if args.fixture:
        common_flags.append("--fixture")
    if args.smoke:
        common_flags.append("--smoke")
    if args.thin_client:
        common_flags.append("--thin-client")
    if args.force:
        common_flags.append("--force")
    if args.config:
        common_flags.extend(["--config", args.config])


    TOTAL_STAGES = 16

    # 0. Preflight
    run_step(1, TOTAL_STAGES, "Preflight Environment & Resource Audit", [py_exe, "-m", "pipeline.00c_preflight"] + common_flags)

    # 1. Profile Dataset
    run_step(2, TOTAL_STAGES, "Dataset Schema & Date Distribution Profiling", [py_exe, "-m", "pipeline.00_profile_dataset"] + common_flags)

    # 2. Parse & Normalize
    parse_cmd = [py_exe, "-m", "pipeline.01_parse"] + common_flags
    if args.max_domains is not None:
        parse_cmd.extend(["--limit", str(args.max_domains)])
    run_step(3, TOTAL_STAGES, "Stream Parsing, Key Resolution & Normalization", parse_cmd)

    # 3. Shortcut Audit
    run_step(4, TOTAL_STAGES, "Shortcut & Single-Feature Leakage Audit", [py_exe, "-m", "pipeline.00b_audit_shortcuts"] + common_flags)

    # 4. Build Graph
    graph_cmd = [py_exe, "-m", "pipeline.02_build_graph"] + common_flags
    if args.max_domains is not None:
        graph_cmd.extend(["--max-domains", str(args.max_domains)])
    run_step(5, TOTAL_STAGES, "Heterogeneous Graph (HDG) Construction", graph_cmd)

    # 5. Baselines
    base_cmd = [py_exe, "-m", "pipeline.03_run_baselines"] + common_flags
    if args.single_seed is not None:
        base_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture or args.smoke:
        base_cmd.extend(["--single-seed", "42"])
    run_step(6, TOTAL_STAGES, "Tabular & Lexical Baselines Evaluation", base_cmd)

    # 6. GNN Evaluation (Random, Time, Group splits)
    epochs_val = args.epochs if args.epochs is not None else (3 if (args.fixture or args.smoke) else 30)
    split_meta = [
        ("Random Split", "random"),
        ("Time Split (Temporal Holdout)", "time"),
        ("Group Split (Disjoint Infrastructure)", "group")
    ]
    step_idx = 7
    for label, s_type in split_meta:
        gnn_cmd = [
            py_exe, "-m", "pipeline.04_train_gnn",
            "--split-type", s_type,
            "--variant", "both",
            "--epochs", str(epochs_val)
        ] + common_flags
        if args.single_seed is not None:
            gnn_cmd.extend(["--single-seed", str(args.single_seed)])
        elif args.fixture or args.smoke:
            gnn_cmd.extend(["--single-seed", "42"])
        run_step(step_idx, TOTAL_STAGES, f"HeteroGNN Training & Testing [{label}]", gnn_cmd)
        step_idx += 1

    # 7. Ablation Study
    abl_cmd = [
        py_exe, "-m", "pipeline.05_ablation",
        "--epochs", str(epochs_val)
    ] + common_flags
    if args.single_seed is not None:
        abl_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture or args.smoke:
        abl_cmd.extend(["--single-seed", "42"])
    run_step(10, TOTAL_STAGES, "Systematic Graph Ablation Study", abl_cmd)

    # 8. Structural Attack
    atk_cmd = [
        py_exe, "-m", "pipeline.06_attack_eval",
        "--epochs", str(epochs_val)
    ] + common_flags
    run_step(11, TOTAL_STAGES, "Adversarial Structural Attack & GNNGuard Recovery", atk_cmd)

    # 9. Latency Benchmark
    lat_cmd = [
        py_exe, "-m", "pipeline.07_latency_eval",
        "--num-queries", "20" if (args.fixture or args.smoke) else "500",
        "--warmup", "5" if (args.fixture or args.smoke) else "20"
    ] + common_flags
    run_step(12, TOTAL_STAGES, "Operational CPU Inference Latency Benchmark", lat_cmd)

    # 10. Provenance Checks
    run_step(13, TOTAL_STAGES, "Dataset Provenance & Cross-Source Generalization Audit", [py_exe, "-m", "pipeline.08_provenance_checks"] + common_flags)

    # 11. Streaming Continual Learning
    stream_cmd = [py_exe, "-m", "pipeline.09_streaming_eval"] + common_flags
    if args.fixture or args.smoke:
        stream_cmd.extend(["--max-months", "3"])
    run_step(14, TOTAL_STAGES, "Streaming Continual Learning & Online Adaptation", stream_cmd)

    # 12. BIND RPZ Rule Emission
    run_step(15, TOTAL_STAGES, "BIND Response Policy Zone (RPZ) Firewall Rule Emission", [py_exe, "-m", "pipeline.10_emit_rpz"] + common_flags)

    # 13. Comprehensive Report Generation
    run_step(16, TOTAL_STAGES, "Comprehensive Evaluation Report Compilation", [py_exe, "-m", "pipeline.11_make_report"] + common_flags)

    print(f"\n{pfx}==================================================")
    print(f"{pfx}Malicious-HDG complete pipeline finished successfully!")
    print(f"{pfx}All artifacts generated in: {paths.results_dir}")
    print(f"{pfx}Comprehensive report: {paths.results_dir / 'REPORT.md'}")
    print(f"{pfx}==================================================")


if __name__ == "__main__":
    main()
