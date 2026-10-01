#!/usr/bin/env python3
"""
CLI script to construct cumulative heterogeneous base graph.
Saves heterodata.pt (base graph with raw feature matrices and topology) and id_maps.json.
Per-split feature standardization is performed downstream per evaluation seed to eliminate leakage.
Supports --max-domains N and --smoke.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from src.hdg.config import load_config, get_resolved_paths, init_thread_pool
from src.hdg.data.graph import build_base_graph, apply_split_scaling


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Heterogeneous Base Graph.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains for fast test")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)
    paths.check_write_path(paths.processed_dir)

    parquet_file = paths.processed_dir / "domains.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Processed domains file not found: {parquet_file}. Run 01_parse.py first.")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    max_d = args.max_domains or (5000 if args.smoke else None)
    if max_d is not None and len(df) > max_d:
        print(f"{pfx}Subsampling to {max_d} domains...")
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    print(f"{pfx}Building HeteroData base graph (raw features + topology)...")
    base_graph = build_base_graph(df, config=cfg)

    # For standard inspecting/compat, also provide unscaled x if needed
    if not hasattr(base_graph["domain"], "x") or base_graph["domain"].x is None:
        base_graph["domain"].x = base_graph["domain"].raw_x.clone()

    for nt in ["ip", "nameserver", "registrar", "asn", "certificate"]:
        if nt in base_graph.node_types and (not hasattr(base_graph[nt], "x") or base_graph[nt].x is None):
            base_graph[nt].x = torch.zeros((base_graph[nt].num_nodes, 1), dtype=torch.float)

    # Save base graph artifact
    graph_pt_path = paths.processed_dir / "heterodata.pt"
    torch.save(base_graph, graph_pt_path)

    id_maps_path = paths.processed_dir / "id_maps.json"
    with open(id_maps_path, "w", encoding="utf-8") as f:
        serializable_maps = {
            nt: {str(k): v for k, v in m.items()}
            for nt, m in getattr(base_graph, "id_maps", {}).items()
        }
        json.dump(serializable_maps, f, indent=2)

    print(f"\n{pfx}=== BASE GRAPH CONSTRUCTION COMPLETE ===")
    print(f"Graph file: {graph_pt_path}")
    print(f"Node types: {base_graph.node_types}")
    print(f"Edge types: {len(base_graph.edge_types)} relational edge types")
    for nt in base_graph.node_types:
        dim = base_graph[nt].x.shape[1] if hasattr(base_graph[nt], "x") and base_graph[nt].x is not None else 0
        print(f"  - {nt}: {base_graph[nt].num_nodes} nodes (dim: {dim})")


if __name__ == "__main__":
    main()
