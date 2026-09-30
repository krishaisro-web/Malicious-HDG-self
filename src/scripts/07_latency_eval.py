#!/usr/bin/env python3
"""
CLI script to measure per-query inference latency on CPU for Malicious-HDG.
Benchmarks:
1. Feature lookup & extraction
2. Ego-subgraph / model forward pass
3. End-to-end query latency
Outputs to results/tables/latency_results.json and latency_results.md.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from models.hetero_gnn import HeteroGNN
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.graph import TrainOnlyStandardizer
from src.hdg.latency import benchmark_cpu_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU Query Latency.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of test queries to measure")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    scaler_file = paths.processed_dir / "scaler_stats.json"

    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError("Processed artifacts not found. Run 01_parse.py and 02_build_graph.py first.")

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading graph and scaler stats...")
    df = pd.read_parquet(parquet_file)
    data = torch.load(graph_file, weights_only=False)

    scaler = TrainOnlyStandardizer()
    if scaler_file.exists():
        with open(scaler_file, "r", encoding="utf-8") as f:
            s_dict = json.load(f)
            scaler.mean = np.array(s_dict["mean"])
            scaler.std = np.array(s_dict["std"])

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage"
    )

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Benchmarking CPU latency over {args.num_queries} queries (warmup: {args.warmup})...")
    latency_stats = benchmark_cpu_latency(
        model=model,
        data=data,
        scaler=scaler,
        df=df,
        num_queries=args.num_queries,
        warmup=args.warmup
    )

    # Save JSON
    json_path = out_dir / "latency_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=2)

    # Save Markdown
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}CPU Inference Latency Benchmark",
        f"*Hardware: CPU, Sample: {latency_stats['num_queries_benchmarked']} queries*",
        "",
        "| Processing Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "|---|---|---|---|---|---|"
    ]
    for stage_name, s_key in [("Feature Extraction & Standardization", "feature_lookup"),
                               ("Model Forward Pass", "forward_pass"),
                               ("End-to-End Query Latency", "end_to_end")]:
        st = latency_stats[s_key]
        md_lines.append(
            f"| {stage_name} | {st['mean_ms']:.3f} | {st['median_ms']:.3f} | {st['p95_ms']:.3f} | {st['min_ms']:.3f} | {st['max_ms']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("> Protocol details available in `docs/latency.md`.")

    md_path = out_dir / "latency_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== LATENCY BENCHMARK COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")
    e2e = latency_stats["end_to_end"]
    print(f"End-to-End Latency: Mean = {e2e['mean_ms']:.2f} ms, P95 = {e2e['p95_ms']:.2f} ms")


if __name__ == "__main__":
    main()
