#!/usr/bin/env python3
"""
CLI script to run ablation experiments on Malicious-HDG.
Ablations evaluated:
1. Full Graph (SAGE)
2. Without Certificate nodes & edges (evaluates value of certificate infrastructure)
3. Without Nameserver edges
4. Without Registrar edges
5. Without ASN edges
6. SAGE vs Attn
7. Graph-only vs Graph + Lexical features
Saves ablation_results.json and ablation_results.md.
"""

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from models.hetero_gnn import HeteroGNN
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.graph import build_hetero_graph
from src.hdg.splits import random_stratified_split
from src.hdg.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ablation Experiments.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Domains file not found: {parquet_file}")

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    seeds = [args.single_seed] if args.single_seed is not None else cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])
    train_epochs = args.epochs if args.epochs is not None else cfg.get("model", {}).get("training", {}).get("epochs", 30)

    ablation_configs = {
        "full_sage": {"include_certs": True, "use_lexical": True, "drop_edge": None, "variant": "sage"},
        "no_certificates": {"include_certs": False, "use_lexical": True, "drop_edge": None, "variant": "sage"},
        "no_nameservers": {"include_certs": True, "use_lexical": True, "drop_edge": "uses_ns", "variant": "sage"},
        "no_registrars": {"include_certs": True, "use_lexical": True, "drop_edge": "registered_by", "variant": "sage"},
        "no_asn": {"include_certs": True, "use_lexical": True, "drop_edge": "belongs_to_asn", "variant": "sage"},
        "graph_only_no_lexical": {"include_certs": True, "use_lexical": False, "drop_edge": None, "variant": "sage"},
        "attn_variant": {"include_certs": True, "use_lexical": True, "drop_edge": None, "variant": "attn"},
    }

    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ablation_configs.keys()}

    for name, a_cfg in ablation_configs.items():
        print(f"\n=== Ablation: {name} ===")
        for s in seeds:
            splits = random_stratified_split(df["label"].to_numpy(), seed=s)
            data, _, _ = build_hetero_graph(
                df=df,
                train_indices=splits.train_indices,
                use_lexical=a_cfg["use_lexical"],
                include_certificates=a_cfg["include_certs"]
            )

            # Drop specified edge if configured
            if a_cfg["drop_edge"] == "uses_ns":
                del data["domain", "uses_ns", "nameserver"]
                del data["nameserver", "rev_uses_ns", "domain"]
            elif a_cfg["drop_edge"] == "registered_by":
                del data["domain", "registered_by", "registrar"]
                del data["registrar", "rev_registered_by", "domain"]
            elif a_cfg["drop_edge"] == "belongs_to_asn":
                del data["ip", "belongs_to_asn", "asn"]
                del data["asn", "rev_belongs_to_asn", "ip"]

            in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
            model = HeteroGNN(
                metadata=data.metadata(),
                in_channels_dict=in_channels,
                variant=a_cfg["variant"]
            )

            test_eval, _, _, _ = train_eval_gnn(
                model=model,
                data=data,
                splits=splits,
                df=df,
                epochs=train_epochs,
                seed=s
            )
            results[name].append(test_eval)
            print(f"  Seed {s} -> ROC-AUC: {test_eval['roc_auc']:.4f}, F1: {test_eval['f1']:.4f}")

    aggregated: Dict[str, Any] = {}
    for name, runs in results.items():
        aucs = [r["roc_auc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        aggregated[name] = {
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "runs": runs
        }

    # Save JSON
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Save Markdown
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}Ablation Study Results",
        f"*Evaluated across {len(seeds)} seeded runs (mean ± std)*",
        "",
        "| Ablation Condition | ROC-AUC | F1 Score | AUC Diff vs Full |",
        "|---|---|---|---|"
    ]
    baseline_auc = aggregated["full_sage"]["roc_auc_mean"]
    for name, stat in aggregated.items():
        diff = stat["roc_auc_mean"] - baseline_auc
        md_lines.append(
            f"| `{name}` | {stat['roc_auc_mean']:.4f} ± {stat['roc_auc_std']:.4f} | {stat['f1_mean']:.4f} ± {stat['f1_std']:.4f} | {diff:+.4f} |"
        )

    md_path = out_dir / "ablation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== ABLATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
