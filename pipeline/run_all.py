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


def run_cmd(cmd: List[str]) -> None:
    print(f"\n[RUN] Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


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


    # 0. Preflight
    preflight_cmd = [py_exe, "-m", "pipeline.00c_preflight"] + common_flags
    run_cmd(preflight_cmd)

    # 1. Profile Dataset
    profile_cmd = [py_exe, "-m", "pipeline.00_profile_dataset"]
    if args.fixture:
        profile_cmd.append("--fixture")
    run_cmd(profile_cmd)

    # 2. Parse & Normalize
    parse_cmd = [py_exe, "-m", "pipeline.01_parse"]
    if args.fixture:
        parse_cmd.append("--fixture")
    if args.max_domains is not None:
        parse_cmd.extend(["--limit", str(args.max_domains)])
    run_cmd(parse_cmd)

    # 3. Shortcut Audit
    audit_cmd = [py_exe, "-m", "pipeline.00b_audit_shortcuts"]
    if args.fixture:
        audit_cmd.append("--fixture")
    run_cmd(audit_cmd)

    # 4. Build Graph
    graph_cmd = [py_exe, "-m", "pipeline.02_build_graph"]
    if args.fixture:
        graph_cmd.append("--fixture")
    if args.max_domains is not None:
        graph_cmd.extend(["--max-domains", str(args.max_domains)])
    run_cmd(graph_cmd)

    # 5. Baselines
    base_cmd = [py_exe, "-m", "pipeline.03_run_baselines"] + common_flags
    if args.single_seed is not None:
        base_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture or args.smoke:
        base_cmd.extend(["--single-seed", "42"])
    run_cmd(base_cmd)

    # 6. GNN Evaluation
    epochs_val = args.epochs if args.epochs is not None else (3 if (args.fixture or args.smoke) else 30)
    gnn_cmd = [
        py_exe, "-m", "pipeline.04_train_gnn",
        "--variant", "both",
        "--epochs", str(epochs_val)
    ] + common_flags
    if args.single_seed is not None:
        gnn_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture or args.smoke:
        gnn_cmd.extend(["--single-seed", "42"])
    run_cmd(gnn_cmd)

    # 7. Ablation Study
    abl_cmd = [
        py_exe, "-m", "pipeline.05_ablation",
        "--epochs", str(epochs_val)
    ] + common_flags
    if args.single_seed is not None:
        abl_cmd.extend(["--single-seed", str(args.single_seed)])
    elif args.fixture or args.smoke:
        abl_cmd.extend(["--single-seed", "42"])
    run_cmd(abl_cmd)

    # 8. Structural Attack
    atk_cmd = [
        py_exe, "-m", "pipeline.06_attack_eval",
        "--epochs", str(epochs_val)
    ] + common_flags
    run_cmd(atk_cmd)

    # 9. Latency Benchmark
    lat_cmd = [
        py_exe, "-m", "pipeline.07_latency_eval",
        "--num-queries", "20" if (args.fixture or args.smoke) else "500",
        "--warmup", "5" if (args.fixture or args.smoke) else "20"
    ] + common_flags
    run_cmd(lat_cmd)

    # 10. Provenance Checks
    prov_cmd = [py_exe, "-m", "pipeline.08_provenance_checks"] + common_flags
    run_cmd(prov_cmd)

    # 11. Streaming Continual Learning
    stream_cmd = [py_exe, "-m", "pipeline.09_streaming_eval"] + common_flags
    if args.fixture or args.smoke:
        stream_cmd.extend(["--max-months", "3"])
    run_cmd(stream_cmd)

    # 12. BIND RPZ Rule Emission
    rpz_cmd = [py_exe, "-m", "pipeline.10_emit_rpz"] + common_flags
    run_cmd(rpz_cmd)

    # 13. Comprehensive Report Generation
    report_cmd = [py_exe, "-m", "pipeline.11_make_report"] + common_flags
    run_cmd(report_cmd)

    print(f"\n{pfx}==================================================")
    print(f"{pfx}Malicious-HDG complete pipeline finished successfully!")
    print(f"{pfx}All artifacts generated in: {paths.results_dir}")
    print(f"{pfx}Comprehensive report: {paths.results_dir / 'REPORT.md'}")
    print(f"{pfx}==================================================")


if __name__ == "__main__":
    main()
