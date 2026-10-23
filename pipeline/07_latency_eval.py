#!/usr/bin/env python3
"""
CLI script to measure per-query inference latency on CPU for Malicious-HDG.
Benchmarks:
1. Batch size = 1 (single-query lookup) and batch size = 32 (mini-batch)
2. Single-thread mode (1 thread) and multi-thread mode (all available CPU cores)
3. Hardware recording: CPU model, cores, PyTorch threads, PyTorch version, OS
4. Breakdown: feature lookup, 2-hop ego-subgraph construction, model forward, end-to-end
5. Percentiles: mean, median, p95, p99 across >= 500 queries with >= 20 warmup queries
Outputs to results/tables/latency_results.json, latency_results.md, and results/runs/latency_eval.json.
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    should_skip_run,
    save_run_result,
    init_thread_pool,
)
from src.hdg.eval.latency import get_cpu_model_name, benchmark_cpu_query_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CPU Query Latency.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--num-queries", type=int, default=None, help="Number of test queries to measure (default: 500, smoke: 50)")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup iterations (default: 20, smoke: 5)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint .pt")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    run_name = "latency_eval"
    if should_skip_run(paths, run_name, force=args.force):
        print(f"[{run_name}] Run already completed. Skipping (use --force to rerun).")
        return

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"

    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError(
            f"Processed artifacts not found in {paths.processed_dir}. Run 01_parse.py and 02_build_graph.py first."
        )

    cfg_lat = cfg.get("latency", {})
    num_queries = args.num_queries if args.num_queries is not None else cfg_lat.get("num_queries", 50 if args.smoke or args.fixture else 500)
    warmup = args.warmup if args.warmup is not None else cfg_lat.get("warmup", 5 if args.smoke or args.fixture else 20)

    from src.hdg.data.graph import load_aligned_graph_cohort

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading aligned graph and dataset...")
    data, df, manifest = load_aligned_graph_cohort(paths.processed_dir)

    if len(df) != data["domain"].num_nodes:
        raise ValueError(
            "Aligned dataframe size does not match graph domain-node count: "
            f"dataframe={len(df)}, graph={data['domain'].num_nodes}"
        )

    if "_graph_idx" in df.columns:
        df = df.drop(columns=["_graph_idx"])
    print(f"[LATENCY] Aligned {len(df):,} dataframe rows to graph domain-node order.")

    from src.hdg.data.splits import random_stratified_split
    from src.hdg.data.graph import apply_split_scaling

    # Apply training-time split scaling to match production inference inputs
    splits = random_stratified_split(df["label"].to_numpy(), seed=42)
    scaled_data, _ = apply_split_scaling(
        base=data,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        degree_mode=cfg.get("graph", {}).get("degree_mode", "transductive")
    )

    in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
    
    # 1. Identify and load production checkpoint
    ckpt_dir = paths.checkpoints_dir
    candidates = []
    if args.checkpoint:
        cand = Path(args.checkpoint)
        if cand.exists():
            candidates.append(cand)
    else:
        preferred = [
            ckpt_dir / "gnn_sage_random_seed_42.pt",
            ckpt_dir / "gnn_attn_random_seed_42.pt",
            ckpt_dir / "gnn_sage_guard_random_seed_42.pt",
            ckpt_dir / "gnn_sage_time_seed_42.pt",
        ]
        for p in preferred:
            if p.exists() and p not in candidates:
                candidates.append(p)
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.glob("*.pt")):
                if not p.name.endswith("_train_state.pt") and p not in candidates:
                    candidates.append(p)

    model = None
    loaded_ckpt_file = None
    for cand_path in candidates:
        try:
            ckpt_obj = torch.load(cand_path, map_location="cpu", weights_only=False)
            state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
            
            # Strict model_config validation: validate all 8 architectural fields
            ckpt_cfg = ckpt_obj.get("model_config") if isinstance(ckpt_obj, dict) else None
            if not isinstance(ckpt_cfg, dict):
                raise ValueError("Checkpoint is missing required 'model_config' dictionary.")

            required_fields = ["variant", "hidden_dim", "out_dim", "num_layers", "num_heads", "dropout", "prune_threshold", "use_edges"]
            missing_fields = [f for f in required_fields if f not in ckpt_cfg]
            if missing_fields:
                raise ValueError(f"Checkpoint model_config is incomplete; missing required fields: {missing_fields}")

            # 1. Variant
            if not isinstance(ckpt_cfg["variant"], str):
                raise TypeError(f"model_config['variant'] must be str, got {type(ckpt_cfg['variant']).__name__}")
            variant = ckpt_cfg["variant"].strip().lower()
            if variant not in ["sage", "attn", "sage_guard"]:
                raise ValueError(f"Unsupported architecture variant: '{variant}'. Must be one of ['sage', 'attn', 'sage_guard']")

            # 2. Dimensions and Layers (must be strictly int, not bool or str)
            for int_f, min_v in [("hidden_dim", 4), ("out_dim", 2), ("num_layers", 1), ("num_heads", 1)]:
                val = ckpt_cfg[int_f]
                if not isinstance(val, int) or isinstance(val, bool):
                    raise TypeError(f"model_config['{int_f}'] must be int, got {type(val).__name__} ({val!r})")
                if val < min_v:
                    raise ValueError(f"model_config['{int_f}'] must be >= {min_v}, got {val}")

            hidden_dim = ckpt_cfg["hidden_dim"]
            out_dim = ckpt_cfg["out_dim"]
            num_layers = ckpt_cfg["num_layers"]
            num_heads = ckpt_cfg["num_heads"]

            # 3. Dropout and prune_threshold (must be float/int in [0.0, 1.0])
            for float_f in ["dropout", "prune_threshold"]:
                val = ckpt_cfg[float_f]
                if not isinstance(val, (float, int)) or isinstance(val, bool):
                    raise TypeError(f"model_config['{float_f}'] must be float, got {type(val).__name__} ({val!r})")
                if not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"model_config['{float_f}'] must be in [0.0, 1.0], got {val}")

            dropout = float(ckpt_cfg["dropout"])
            prune_threshold = float(ckpt_cfg["prune_threshold"])

            # 4. use_edges (must strictly be bool)
            if not isinstance(ckpt_cfg["use_edges"], bool):
                raise TypeError(f"model_config['use_edges'] must be bool, got {type(ckpt_cfg['use_edges']).__name__} ({ckpt_cfg['use_edges']!r})")
            use_edges = ckpt_cfg["use_edges"]

            cand_model = HeteroGNN(
                metadata=scaled_data.metadata(),
                in_channels_dict=in_channels,
                variant=variant,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                prune_threshold=prune_threshold,
                use_edges=use_edges
            )
            cand_model.load_state_dict(state_dict)
            model = cand_model
            loaded_ckpt_file = cand_path.name
            print(f"[LATENCY] Loaded verified production checkpoint: {cand_path.name} "
                  f"(variant={variant}, hidden_dim={hidden_dim}, out_dim={out_dim}, layers={num_layers}, heads={num_heads})")
            break
        except Exception as e:
            print(f"[LATENCY] Could not load checkpoint {cand_path.name} ({e}), trying next candidate...")

    if model is None:
        print("[LATENCY] No saved checkpoint found. Initializing reference SAGE model...")
        model = HeteroGNN(
            metadata=scaled_data.metadata(),
            in_channels_dict=in_channels,
            variant="sage"
        )

    # Hardware specs
    cpu_model = get_cpu_model_name()
    core_count = os.cpu_count() or 1
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    torch_version = torch.__version__

    max_threads = init_thread_pool(cfg)
    thread_modes = [1]
    if max_threads > 1:
        thread_modes.append(max_threads)

    batch_sizes = [1, 32]
    all_results = []

    original_threads = torch.get_num_threads()

    try:
        for t_count in thread_modes:
            torch.set_num_threads(t_count)
            print(f"\n--- Benchmarking with {t_count} PyTorch Thread(s) ---")
            for bs in batch_sizes:
                print(f"Evaluating Batch Size = {bs} ({num_queries} queries, {warmup} warmup)...")
                stats = benchmark_cpu_query_latency(
                    model=model,
                    data=scaled_data,
                    df=df,
                    batch_size=bs,
                    num_queries=num_queries,
                    warmup=warmup,
                    seed=cfg.get("eval_seeds", [42])[0]
                )
                stats["threads"] = t_count
                all_results.append(stats)
                e2e = stats["end_to_end"]
                print(f"  BS={bs}, Threads={t_count} -> E2E Mean: {e2e['mean_ms']:.2f}ms, P95: {e2e['p95_ms']:.2f}ms, P99: {e2e['p99_ms']:.2f}ms")
    finally:
        torch.set_num_threads(original_threads)

    payload = {
        "hardware": {
            "cpu_model": cpu_model,
            "core_count": core_count,
            "os": os_name,
            "torch_version": torch_version,
            "max_available_threads": max_threads
        },
        "config": {
            "num_queries": num_queries,
            "warmup": warmup,
            "is_fixture": args.fixture,
            "is_smoke": args.smoke,
            "checkpoint_file": loaded_ckpt_file
        },
        "benchmarks": all_results
    }

    # Save JSON
    json_path = out_dir / "latency_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save Markdown Table
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}CPU Inference Latency Benchmark",
        "",
        "## System Environment",
        f"- **CPU Model**: `{cpu_model}`",
        f"- **Physical/Logical Cores**: `{core_count}`",
        f"- **Operating System**: `{os_name}`",
        f"- **PyTorch Version**: `{torch_version}`",
        f"- **Evaluated Checkpoint**: `{loaded_ckpt_file or 'Reference SAGE (fresh)'}`",
        f"- **Queries Measured**: `{num_queries}` (Warmup: `{warmup}`)",
        "",
        "## Benchmark Results (ms per query)",
        "| Batch Size | Threads | Feature Lookup Mean (P95) | 2-Hop Ego Build Mean (P95) | Model Forward Mean (P95) | End-to-End Mean | Median | P95 | P99 |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for b in all_results:
        bs = b["batch_size"]
        th = b["threads"]
        feat = f"{b['feature_lookup']['mean_ms']:.2f} ({b['feature_lookup']['p95_ms']:.2f})"
        ego = f"{b['ego_subgraph_build']['mean_ms']:.2f} ({b['ego_subgraph_build']['p95_ms']:.2f})"
        fwd = f"{b['model_forward']['mean_ms']:.2f} ({b['model_forward']['p95_ms']:.2f})"
        e2e = b["end_to_end"]
        md_lines.append(
            f"| {bs} | {th} | {feat} | {ego} | {fwd} | **{e2e['mean_ms']:.2f}** | {e2e['median_ms']:.2f} | {e2e['p95_ms']:.2f} | {e2e['p99_ms']:.2f} |"
        )

    md_lines.append("")
    md_lines.append("> Detailed methodology and ego-subgraph mathematical equivalence documented in `docs/latency.md`.")

    md_path = out_dir / "latency_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Resumability record
    save_run_result(paths, run_name, payload)

    print(f"\n{pfx}=== LATENCY BENCHMARK COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
