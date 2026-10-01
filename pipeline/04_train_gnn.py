#!/usr/bin/env python3
"""
CLI script to train and evaluate HeteroGNN models (SAGE, Attn, and SAGE_Guard variants)
across multiple evaluation seeds and split methodologies (random, time, group).
Computes exact McNemar tests and paired bootstrap AUC differences across all seeds.
Outputs tables and run summaries to results/runs/ and results/tables/.
Supports --smoke, --max-domains N, and --force for resumability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    should_skip_run,
    save_run_result
)
from src.hdg.data.graph import apply_split_scaling
from src.hdg.metrics import (
    multi_seed_paired_comparisons,
    compute_bootstrap_cis
)
from src.hdg.data.splits import (
    random_stratified_split,
    rolling_origin_temporal_splits,
    group_split_bipartite,
    group_split_by_asn,
    DataSplits
)
from src.hdg.training.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate HeteroGNN models.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--variant", type=str, default="both", choices=["sage", "attn", "sage_guard", "both", "all"], help="GNN variant")
    parser.add_argument("--split-type", type=str, default="random", choices=["random", "time", "group"], help="Split type")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached runs")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    tables_dir = paths.results_dir / "tables"
    logs_dir = paths.results_dir / "logs"
    ckpt_dir = paths.checkpoints_dir
    paths.check_write_path(tables_dir)
    paths.check_write_path(ckpt_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError("Processed data not found. Run 01_parse.py and 02_build_graph.py first.")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading graph and domain metadata...")
    df = pd.read_parquet(parquet_file)
    max_d = args.max_domains or (5000 if args.smoke else None)
    if max_d is not None and len(df) > max_d:
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    base_graph = torch.load(graph_file, weights_only=False)

    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.smoke:
        seeds = [42]
    else:
        seeds = cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])

    if args.variant == "all":
        variants = ["sage", "attn", "sage_guard"]
    elif args.variant == "both":
        variants = ["sage", "attn"]
    else:
        variants = [args.variant]

    if args.epochs is not None:
        train_epochs = args.epochs
    elif args.smoke or args.fixture:
        train_epochs = 3
    else:
        train_epochs = cfg.get("model", {}).get("training", {}).get("epochs", 200)

    patience = 2 if (args.smoke or args.fixture) else cfg.get("model", {}).get("training", {}).get("patience", 20)
    boot_samples = 100 if (args.smoke or args.fixture) else cfg.get("evaluation", {}).get("bootstrap_samples", 1000)

    print(f"{pfx}Evaluating GNN on split: '{args.split_type}' across seeds: {seeds}...")

    results_by_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    predictions_by_variant: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {v: {} for v in variants}

    for var in variants:
        print(f"\n================ Running Variant: {var.upper()} ================")
        var_cfg = cfg.get("model", {}).get(var, {})
        hidden_dim = var_cfg.get("hidden_dim", 64)
        out_dim = var_cfg.get("out_dim", 32)
        num_layers = var_cfg.get("num_layers", 2)
        num_heads = var_cfg.get("num_heads", 4)
        dropout = var_cfg.get("dropout", 0.2)
        prune_thresh = var_cfg.get("prune_threshold", 0.1)

        for s in seeds:
            run_name = f"gnn_{var}_{args.split_type}_seed_{s}"
            print(f"\n--- Seed {s} ({var}) [{args.split_type}] ---")

            if should_skip_run(paths.results_dir, run_name, force=args.force):
                print(f"  [RESUME] Skipping already completed run: {run_name}")
                with open(paths.results_dir / "runs" / f"{run_name}.json", "r", encoding="utf-8") as f:
                    cached_res = json.load(f)
                results_by_variant[var].append(cached_res["metrics"])
                predictions_by_variant[var][s] = {
                    "y_true": np.array(cached_res["predictions"]["y_true"]),
                    "y_prob": np.array(cached_res["predictions"]["y_prob"]),
                    "y_pred": np.array(cached_res["predictions"]["y_pred"])
                }
                continue

            # 1. Create split for this seed
            asn_fallback_meta = None
            if args.split_type == "random":
                splits = random_stratified_split(df["label"].to_numpy(), seed=s)
            elif args.split_type == "time":
                # Rolling-origin evaluation
                tm_file = paths.processed_dir / "domains_timematched.parquet"
                time_df = pd.read_parquet(tm_file) if tm_file.exists() else df
                transitions = rolling_origin_temporal_splits(
                    time_df,
                    min_distinct_months=cfg.get("temporal", {}).get("min_distinct_months", 3),
                    min_per_class_per_window=cfg.get("temporal", {}).get("min_per_class_per_window", 200),
                    seed=s
                )
                if not transitions:
                    print("  [WARN] No valid rolling-origin transitions available; skipping time split.")
                    continue
                # Evaluate on the primary transition
                trans_name, splits = transitions[0]
                print(f"  Rolling-Origin Transition: {trans_name}")
            elif args.split_type == "group":
                joint_grp = cfg.get("group", {}).get("joint_grouping", False)
                splits, grp_meta = group_split_bipartite(df, joint_grouping=joint_grp, seed=s)
                print(f"  Giant component fraction: {grp_meta['giant_component_fraction']:.4f}")
                if grp_meta["giant_component_fraction"] > 0.50:
                    print("  [INFO] Giant component > 50%; providing group split by primary ASN...")
                    splits, asn_meta = group_split_by_asn(df, seed=s)
                    asn_fallback_meta = asn_meta.get("leakage_report")
            else:
                splits = random_stratified_split(df["label"].to_numpy(), seed=s)

            # 2. Scale features per split to eliminate leakage
            scaled_data, scaler_stats = apply_split_scaling(
                base=base_graph,
                train_indices=splits.train_indices,
                val_indices=splits.val_indices,
                degree_mode=cfg.get("graph", {}).get("degree_mode", "transductive")
            )

            # 3. Instantiate Model
            in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
            model = HeteroGNN(
                metadata=scaled_data.metadata(),
                in_channels_dict=in_channels,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                variant=var,
                prune_threshold=prune_thresh
            )

            ckpt_path = ckpt_dir / f"{run_name}.pt"
            log_path = logs_dir / f"{run_name}.jsonl"

            # 4. Train & Evaluate
            test_eval, y_test, y_probs, _ = train_eval_gnn(
                model=model,
                data=scaled_data,
                splits=splits,
                df=df,
                epochs=train_epochs,
                lr=cfg.get("model", {}).get("training", {}).get("lr", 0.005),
                lr_grid=cfg.get("model", {}).get("training", {}).get("lr_grid", [0.001, 0.003, 0.005]),
                patience=patience,
                bootstrap_samples=boot_samples,
                checkpoint_path=ckpt_path,
                log_file=log_path,
                seed=s
            )

            if asn_fallback_meta:
                test_eval["asn_leakage_report"] = asn_fallback_meta

            y_pred = (y_probs >= test_eval["threshold"]).astype(int)
            results_by_variant[var].append(test_eval)
            predictions_by_variant[var][s] = {
                "y_true": y_test,
                "y_prob": y_probs,
                "y_pred": y_pred
            }

            # Persist run JSON for resumability
            run_payload = {
                "experiment": "gnn_evaluation",
                "variant": var,
                "split_type": args.split_type,
                "seed": s,
                "metrics": test_eval,
                "predictions": {
                    "y_true": y_test.tolist(),
                    "y_prob": y_probs.tolist(),
                    "y_pred": y_pred.tolist()
                }
            }
            save_run_result(paths.results_dir, run_name, run_payload)
            print(f"  Test ROC-AUC: {test_eval['roc_auc']:.4f}, PR-AUC: {test_eval['pr_auc']:.4f}, F1: {test_eval['f1']:.4f}")

    # Aggregate statistics
    aggregated_gnn: Dict[str, Any] = {}
    for var, runs in results_by_variant.items():
        if not runs:
            continue
        aucs = [r["roc_auc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        prs = [r["pr_auc"] for r in runs]
        tpr_01 = [r.get("operating_points", {}).get("tpr_at_0.1pct_fpr", {}).get("tpr", 0.0) for r in runs]
        aggregated_gnn[var] = {
            "num_seeds": len(runs),
            "split_type": args.split_type,
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_std": round(float(np.std(prs)), 4),
            "tpr_at_0.1pct_fpr_mean": round(float(np.mean(tpr_01)), 4),
            "runs": runs
        }

    # Paired comparisons across all seeds
    paired_comparison_results = {}
    if "sage" in variants and "attn" in variants and len(seeds) > 0:
        paired_comparison_results["sage_vs_attn"] = multi_seed_paired_comparisons(
            seeds=seeds,
            predictions_a=predictions_by_variant["sage"],
            predictions_b=predictions_by_variant["attn"],
            n_bootstrap=boot_samples
        )

    if "sage" in variants and "sage_guard" in variants and len(seeds) > 0:
        paired_comparison_results["sage_vs_sage_guard"] = multi_seed_paired_comparisons(
            seeds=seeds,
            predictions_a=predictions_by_variant["sage"],
            predictions_b=predictions_by_variant["sage_guard"],
            n_bootstrap=boot_samples
        )

    final_report = {
        "is_fixture": args.fixture,
        "is_smoke": args.smoke,
        "split_type": args.split_type,
        "variants": aggregated_gnn,
        "paired_comparisons": paired_comparison_results
    }

    # Save JSON
    json_path = tables_dir / f"gnn_evaluation_{args.split_type}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Save Markdown
    pfx_t = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx_t}HeteroGNN Evaluation ({args.split_type.upper()} Split)",
        f"*Evaluated across {len(seeds)} seeded runs (mean ± std)*",
        "",
        "| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |",
        "|---|---|---|---|---|"
    ]
    for var, stat in aggregated_gnn.items():
        md_lines.append(
            f"| `HeteroGNN ({var.upper()})` | {stat['roc_auc_mean']:.4f} ± {stat['roc_auc_std']:.4f} | {stat['f1_mean']:.4f} ± {stat['f1_std']:.4f} | {stat['pr_auc_mean']:.4f} ± {stat['pr_auc_std']:.4f} | {stat['tpr_at_0.1pct_fpr_mean']:.4f} |"
        )
    md_lines.append("")

    if paired_comparison_results:
        md_lines.append("## Paired Hypothesis Testing (Across All Seeds)")
        for pair_name, comp_data in paired_comparison_results.items():
            pooled = comp_data.get("pooled_summary", {})
            mcn = pooled.get("mcnemar", {})
            diff = pooled.get("bootstrap_auc_diff", {})
            md_lines.append(f"### Comparison: `{pair_name}`")
            md_lines.append(f"- **Pooled Exact McNemar Statistic (min(b,c))**: {mcn.get('exact_statistic_min_b_c', 'N/A')} (p = {mcn.get('pvalue', 'N/A')})")
            md_lines.append(f"- **Pooled Paired Bootstrap AUC Difference**: {diff.get('mean_diff', 'N/A')} (95% CI: [{diff.get('ci_lower', 'N/A')}, {diff.get('ci_upper', 'N/A')}], p = {diff.get('pvalue_two_sided', 'N/A')})")
            md_lines.append("")

    md_path = tables_dir / f"gnn_evaluation_{args.split_type}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== GNN EVALUATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
