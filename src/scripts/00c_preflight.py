#!/usr/bin/env python3
"""
Preflight environment verification and dry-run extrapolation for Malicious-HDG.
Validates dependencies, CPU/RAM/disk resources, raw dataset files,
and estimates training time with a stratified 10% subsample benchmark.
Fails loudly with actionable guidance if requirements are not met.
"""

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths, init_thread_pool


def check_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Verifies that all required third-party libraries import correctly."""
    required = [
        "torch",
        "torch_geometric",
        "xgboost",
        "sklearn",
        "statsmodels",
        "pyarrow",
        "tldextract",
        "ijson"
    ]
    status = {}
    for pkg in required:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            status[pkg] = (True, version)
        except ImportError as e:
            status[pkg] = (False, str(e))
    return status


def get_ram_info() -> Tuple[float, float]:
    """Returns (total_ram_gb, available_ram_gb). Uses psutil or falls back to /proc/meminfo / Windows ctypes."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.total / (1024**3), 2), round(vm.available / (1024**3), 2)
    except Exception:
        pass

    # Linux /proc/meminfo fallback
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        try:
            total_kb, avail_kb = 0, 0
            with open(meminfo_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
            return round(total_kb / (1024**2), 2), round(avail_kb / (1024**2), 2)
        except Exception:
            pass

    # Windows fallback
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024**3), 2), round(stat.ullAvailPhys / (1024**3), 2)
        except Exception:
            pass

    return 0.0, 0.0


def run_subsample_benchmark(
    graph_path: Path,
    parquet_path: Path,
    configured_epochs: int,
    sample_ratio: float = 0.10
) -> Dict[str, Any]:
    """Performs a 1-epoch timing of SAGE on a stratified subsample and extrapolates full training time."""
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.hdg.models.hetero_gnn import HeteroGNN

    df = pd.read_parquet(parquet_path)
    data = torch.load(graph_path, weights_only=False)

    n_domains = data["domain"].num_nodes
    y = data["domain"].y.numpy()

    # Stratified subsample
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - sample_ratio), random_state=42)
    sample_idx, _ = next(sss.split(np.arange(n_domains), y))
    sample_tensor = torch.tensor(sample_idx, dtype=torch.long)

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=64,
        out_dim=32,
        num_layers=2,
        variant="sage"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Measure 1 epoch
    model.train()
    optimizer.zero_grad()
    t0 = time.perf_counter()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[sample_tensor], data["domain"].y[sample_tensor])
    loss.backward()
    optimizer.step()
    t1 = time.perf_counter()

    epoch_sec = t1 - t0
    # Linear extrapolation
    est_full_epoch_sec = epoch_sec * (1.0 / sample_ratio)
    est_total_run_sec = est_full_epoch_sec * configured_epochs

    return {
        "subsample_size": len(sample_idx),
        "total_domains": n_domains,
        "subsample_epoch_sec": round(epoch_sec, 3),
        "extrapolated_full_epoch_sec": round(est_full_epoch_sec, 3),
        "configured_epochs": configured_epochs,
        "extrapolated_run_minutes": round(est_total_run_sec / 60.0, 2),
        "extrapolated_run_hours": round(est_total_run_sec / 3600.0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight system check and workload extrapolation.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
    parser.add_argument("--force", action="store_true", help="Force rerun")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)
    num_threads = init_thread_pool()

    pfx = "[FIXTURE] " if args.fixture else "[REAL] "
    print(f"\n{pfx}==================================================")
    print(f"{pfx}MALICIOUS-HDG PREFLIGHT ENVIRONMENT & DATA AUDIT")
    print(f"{pfx}==================================================")

    # 1. System Hardware Resources
    cpu_count = os.cpu_count() or 1
    total_ram, avail_ram = get_ram_info()
    disk_total, disk_used, disk_free = shutil.disk_usage(REPO_ROOT)
    disk_free_gb = round(disk_free / (1024**3), 2)

    print("\n--- System & Hardware Audit ---")
    print(f"  OS Platform:        {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version:     {platform.python_version()} ({sys.executable})")
    print(f"  CPU Cores:          {cpu_count}")
    print(f"  PyTorch Threads:    {num_threads} (set via HDG_THREADS or cpu_count)")
    print(f"  System RAM:         {total_ram:.1f} GB total, {avail_ram:.1f} GB available")
    print(f"  Free Workspace Disk: {disk_free_gb:.1f} GB")

    if disk_free_gb < 5.0:
        print(f"\n[WARNING] Low free disk space: {disk_free_gb} GB (< 5 GB). Processed files & models may require additional space.")

    # 2. Dependency Audit
    print("\n--- Python Package Dependencies ---")
    deps = check_dependencies()
    missing_deps: List[str] = []
    for pkg, (ok, ver_or_err) in deps.items():
        if ok:
            print(f"  [OK]   {pkg.ljust(18)} (v{ver_or_err})")
        else:
            print(f"  [FAIL] {pkg.ljust(18)} (ERROR: {ver_or_err})")
            missing_deps.append(pkg)

    if missing_deps:
        print(f"\n[FATAL] Missing required dependencies: {missing_deps}")
        print("Action required: Run `pip install -r requirements.txt` to install required packages.")
        sys.exit(1)

    # 3. Raw Dataset Files Check
    print("\n--- Raw Dataset Files Check ---")
    expected_files = ["malware.json", "benign_umbrella.json", "benign_cesnet.json"]
    missing_files: List[str] = []
    for fn in expected_files:
        fp = paths.raw_dir / fn
        if fp.exists():
            size_mb = fp.stat().st_size / (1024**2)
            print(f"  [FOUND] {fn.ljust(22)} ({size_mb:.1f} MB) -> {fp}")
        else:
            print(f"  [MISSING] {fn.ljust(22)} -> {fp}")
            missing_files.append(fn)

    if missing_files:
        if args.fixture:
            print("\n[INFO] Fixture raw files missing. Run `python -m src.hdg.fixture` or `run_all --fixture` to generate.")
        else:
            print(f"\n[FATAL] Raw dataset files missing in {paths.raw_dir}: {missing_files}")
            print("Action required: Colleague must place Zenodo DomainRadar v2 JSON files into data_raw/zenodo/ before profiling.")
            sys.exit(1)

    # 4. Processed Artifacts Audit
    parquet_path = paths.processed_dir / "domains.parquet"
    graph_path = paths.processed_dir / "heterodata.pt"

    print("\n--- Processed Artifacts Status ---")
    if not parquet_path.exists() or not graph_path.exists():
        print(f"  [STATUS] Processed artifacts not yet built at: {paths.processed_dir}")
        print("  -> Next step: run `00_profile_dataset.py`, `01_parse.py`, and `02_build_graph.py`.")
    else:
        print(f"  [FOUND] Parquet: {parquet_path}")
        print(f"  [FOUND] Graph:   {graph_path}")

        import pandas as pd
        import torch

        # Validate columns
        df = pd.read_parquet(parquet_path)
        required_cols = {"domain", "label", "source", "t", "no_ip", "no_rdap", "no_tls"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            print(f"\n[FATAL] Parquet missing required schema columns: {missing_cols}")
            sys.exit(1)
        print(f"  [OK] Validated Parquet columns ({len(df.columns)} cols, {len(df)} domains, {int((df['label']==1).sum())} mal / {int((df['label']==0).sum())} ben)")

        # Validate graph
        data = torch.load(graph_path, weights_only=False)
        print(f"  [OK] Graph Node Types: {data.node_types}")
        for nt in data.node_types:
            dim = data[nt].x.shape[1] if hasattr(data[nt], "x") and data[nt].x is not None else 0
            print(f"       - {nt}: {data[nt].num_nodes} nodes (feature dim: {dim})")
        print(f"  [OK] Graph Edge Types: {len(data.edge_types)} relational edge types")

        # 5. Extrapolation Benchmark
        print("\n--- Workload Timing & Linear Extrapolation ---")
        cfg_epochs = cfg.get("model", {}).get("training", {}).get("epochs", 200)
        timing = run_subsample_benchmark(
            graph_path=graph_path,
            parquet_path=parquet_path,
            configured_epochs=cfg_epochs,
            sample_ratio=cfg.get("preflight", {}).get("sample_ratio", 0.10)
        )
        print(f"  Subsample benchmark (10% domain nodes): {timing['subsample_epoch_sec']} s / epoch")
        print(f"  Extrapolated full batch 1-epoch time:   {timing['extrapolated_full_epoch_sec']} s / epoch")
        print(f"  Extrapolated 1-seed ({cfg_epochs} epochs):     ~{timing['extrapolated_run_minutes']} minutes (~{timing['extrapolated_run_hours']} hours)")
        n_seeds = len(cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999]))
        total_seeds_hours = round(timing["extrapolated_run_hours"] * n_seeds, 2)
        print(f"  Extrapolated {n_seeds}-seed full GNN run:       ~{total_seeds_hours} hours total")

    print(f"\n{pfx}==================================================")
    print(f"{pfx}[PASS] Preflight check completed successfully!")
    print(f"{pfx}==================================================\n")


if __name__ == "__main__":
    main()
