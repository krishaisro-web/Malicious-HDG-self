#!/usr/bin/env python3
"""
CLI script to train and evaluate HeteroGNN models (SAGE and Attn variants)
across multiple random seeds and split methodologies (random, time, group).
Computes exact McNemar tests and paired bootstrap AUC differences.
Outputs to results/tables/ and results/figures/.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.hetero_gnn import HeteroGNN
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.metrics import paired_mcnemar_test, paired_bootstrap_auc_difference
from src.hdg.splits import (
    random_stratified_split,
    time_split,
    group_split_bipartite,
    group_split_by_asn,
    DataSplits
)
from src.hdg.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate HeteroGNN.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--variant", type=str, default="sage", choices=["sage", "attn", "both"], help="GNN variant")
    parser.add_argument("--split-type", type=str, default="random", choices=["random", "time", "group"], help="Split type")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    tables_dir = paths.results_dir / "tables"
    figures_dir = paths.results_dir / "figures"
    paths.check_write_path(tables_dir)
    paths.check_write_path(figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError("Processed data not found. Run 01_parse.py and 02_build_graph.py first.")

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading graph and domain metadata...")
    df = pd.read_parquet(parquet_file)
    data = torch.load(graph_file, weights_only=False)

    seeds = [args.single_seed] if args.single_seed is not None else cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])
    variants = ["sage", "attn"] if args.variant == "both" else [args.variant]
    train_epochs = args.epochs if args.epochs is not None else cfg.get("model", {}).get("training", {}).get("epochs", 30)

    # Check time split validity if time split requested
    time_split_valid = False
    ts_file = paths.results_dir / "profile" / "time_split_valid.json"
    if ts_file.exists():
        with open(ts_file, "r", encoding="utf-8") as f:
            time_split_valid = json.load(f).get("time_split_valid", False)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Evaluating GNN on split: '{args.split_type}' across seeds: {seeds}...")

    results_by_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    predictions_by_variant: Dict[str, Dict[str, np.ndarray]] = {}

    for var in variants:
        print(f"\n================ Running Variant: {var.upper()} ================")
        var_cfg = cfg.get("model", {}).get(var, {})
        hidden_dim = var_cfg.get("hidden_dim", 64)
        out_dim = var_cfg.get("out_dim", 32)
        num_layers = var_cfg.get("num_layers", 2)
        num_heads = var_cfg.get("num_heads", 4)
        dropout = var_cfg.get("dropout", 0.2)

        for s in seeds:
            print(f"\n--- Seed {s} ({var}) ---")
            # 1. Create split for this seed
            if args.split_type == "random":
                splits = random_stratified_split(df["label"].to_numpy(), seed=s)
            elif args.split_type == "time":
                splits = time_split(df, time_split_valid=time_split_valid)
            elif args.split_type == "group":
                splits, grp_meta = group_split_bipartite(df, seed=s)
                print(f"  Giant component fraction: {grp_meta['giant_component_fraction']:.4f}")
                if grp_meta["giant_component_fraction"] > 0.50:
                    print("  [INFO] Giant component > 50%; providing group split by primary ASN...")
                    splits = group_split_by_asn(df, seed=s)
            else:
                splits = random_stratified_split(df["label"].to_numpy(), seed=s)

            # 2. Instantiate Model
            in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
            model = HeteroGNN(
                metadata=data.metadata(),
                in_channels_dict=in_channels,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                variant=var
            )

            # 3. Train & Evaluate
            test_eval, y_test, y_probs, _ = train_eval_gnn(
                model=model,
                data=data,
                splits=splits,
                df=df,
                epochs=train_epochs,
                lr=cfg.get("model", {}).get("training", {}).get("lr", 0.005),
                patience=cfg.get("model", {}).get("training", {}).get("patience", 5),
                seed=s
            )

            results_by_variant[var].append(test_eval)
            predictions_by_variant[f"{var}_{s}"] = {
                "y_true": y_test,
                "y_prob": y_probs,
                "y_pred": (y_probs >= test_eval["threshold"]).astype(int)
            }
            print(f"  Test ROC-AUC: {test_eval['roc_auc']:.4f}, PR-AUC: {test_eval['pr_auc']:.4f}, F1: {test_eval['f1']:.4f}")

    # Aggregate statistics
    aggregated_gnn: Dict[str, Any] = {}
    for var, runs in results_by_variant.items():
        aucs = [r["roc_auc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        prs = [r["pr_auc"] for r in runs]
        aggregated_gnn[var] = {
            "num_seeds": len(runs),
            "split_type": args.split_type,
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_std": round(float(np.std(prs)), 4),
            "runs": runs
        }

    # Paired comparisons if both variants run
    paired_comparison_results = {}
    if "sage" in variants and "attn" in variants and len(seeds) > 0:
        first_seed = seeds[0]
        p_sage = predictions_by_variant[f"sage_{first_seed}"]
        p_attn = predictions_by_variant[f"attn_{first_seed}"]

        mcnemar_res = paired_mcnemar_test(
            p_sage["y_true"],
            p_sage["y_pred"],
            p_attn["y_pred"]
        )
        boot_diff = paired_bootstrap_auc_difference(
            p_sage["y_true"],
            p_sage["y_prob"],
            p_attn["y_prob"],
            seed=first_seed
        )
        paired_comparison_results = {
            "mcnemar_sage_vs_attn": mcnemar_res,
            "bootstrap_auc_diff_sage_minus_attn": boot_diff
        }

    final_report = {
        "is_fixture": args.fixture,
        "split_type": args.split_type,
        "variants": aggregated_gnn,
        "paired_comparisons": paired_comparison_results
    }

    # Save JSON
    json_path = tables_dir / f"gnn_evaluation_{args.split_type}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Save Markdown
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}HeteroGNN Evaluation ({args.split_type.upper()} Split)",
        f"*Evaluated across {len(seeds)} seeded runs (mean ± std)*",
        "",
        "| Architecture | ROC-AUC | F1 Score | PR-AUC |",
        "|---|---|---|---|"
    ]
    for var, stat in aggregated_gnn.items():
        md_lines.append(
            f"| `HeteroGNN ({var.upper()})` | {stat['roc_auc_mean']:.4f} ± {stat['roc_auc_std']:.4f} | {stat['f1_mean']:.4f} ± {stat['f1_std']:.4f} | {stat['pr_auc_mean']:.4f} ± {stat['pr_auc_std']:.4f} |"
        )
    md_lines.append("")

    if paired_comparison_results:
        md_lines.append("## Paired Hypothesis Testing (SAGE vs Attn)")
        mcn = paired_comparison_results["mcnemar_sage_vs_attn"]
        diff = paired_comparison_results["bootstrap_auc_diff_sage_minus_attn"]
        md_lines.append(f"- **Exact McNemar Statistic (min(b,c))**: {mcn['exact_statistic_min_b_c']} (p = {mcn['pvalue']:.4f})")
        md_lines.append(f"- **Paired Bootstrap AUC Difference**: {diff['mean_diff']:.4f} (95% CI: [{diff['ci_lower']:.4f}, {diff['ci_upper']:.4f}], p = {diff['pvalue_two_sided']:.4f})")
        md_lines.append("")

    md_path = tables_dir / f"gnn_evaluation_{args.split_type}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== GNN EVALUATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
