#!/usr/bin/env python3
"""
CLI script to construct cumulative heterogeneous graph and persist graph artifacts.
Generates heterodata.pt, id_maps.json, and scaler_stats.json.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.graph import build_hetero_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Heterogeneous Graph.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)
    paths.check_write_path(paths.processed_dir)

    parquet_file = paths.processed_dir / "domains.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Processed domains file not found: {parquet_file}. Run 01_parse.py first.")

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    p_cfg = cfg.get("parsing", {})
    g_cfg = cfg.get("graph", {})
    use_lex = g_cfg.get("features", {}).get("use_lexical", True)
    include_sub = p_cfg.get("include_has_subdomain", False)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Building HeteroData graph...")
    data, scaler, id_maps = build_hetero_graph(
        df=df,
        use_lexical=use_lex,
        include_has_subdomain=include_sub,
        include_certificates=True
    )

    # Save artifacts
    graph_pt_path = paths.processed_dir / "heterodata.pt"
    torch.save(data, graph_pt_path)

    id_maps_path = paths.processed_dir / "id_maps.json"
    with open(id_maps_path, "w", encoding="utf-8") as f:
        # Avoid non-string keys in JSON
        serializable_maps = {
            nt: {str(k): v for k, v in m.items()}
            for nt, m in id_maps.items()
        }
        json.dump(serializable_maps, f, indent=2)

    scaler_path = paths.processed_dir / "scaler_stats.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f, indent=2)

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"\n{pfx}=== GRAPH CONSTRUCTION COMPLETE ===")
    print(f"Graph file: {graph_pt_path}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    for nt in data.node_types:
        print(f"  - {nt}: {data[nt].num_nodes} nodes (dim: {data[nt].x.shape[1]})")
    for et in data.edge_types:
        print(f"  - {et}: {data[et].num_edges} edges")


if __name__ == "__main__":
    main()
