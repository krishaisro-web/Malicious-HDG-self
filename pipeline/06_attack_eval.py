#!/usr/bin/env python3
"""
CLI script to evaluate adversarial robustness of Malicious-HDG under PPT threat model:
Generates 3-configuration table per budget over evaluation seeds:
(A) Clean SAGE (Undefended clean)
(B) Attacked Undefended SAGE (Edge injection + feature manipulation)
(C) Attacked + GNNGuard (SAGE_Guard defense evaluated on attacked graph)
(D) Clean SAGE_Guard (Defense cost on clean graph)
Reports: F1, TPR@0.1% FPR, ROC-AUC, Recall, Evasion Rate, Robustness Delta, and Defense Recovery.
Supports --smoke, --max-domains N, and --force for resumability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.eval.attack import generate_structural_attack, evaluate_attacked_model
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    should_skip_run,
    save_run_result
)
from src.hdg.data.graph import apply_split_scaling
from src.hdg.data.splits import random_stratified_split
from src.hdg.training.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adversarial Structural Attack with GNNGuard Recovery.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached runs")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError("Processed files not found. Run 01_parse.py and 02_build_graph.py first.")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading domain metadata and base graph...")
    df = pd.read_parquet(parquet_file)
    max_d = args.max_domains or (5000 if args.smoke else cfg.get("graph", {}).get("max_domains"))
    if max_d is not None and len(df) > max_d:
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    base_graph = torch.load(graph_file, weights_only=False)

    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.smoke:
        seeds = [42]
    else:
        seeds = cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])

    if args.epochs is not None:
        train_epochs = args.epochs
    elif args.smoke or args.fixture:
        train_epochs = 3
    else:
        train_epochs = cfg.get("model", {}).get("training", {}).get("epochs", 30)

    patience = 2 if (args.smoke or args.fixture) else cfg.get("model", {}).get("training", {}).get("patience", 10)
    budgets = [1, 2] if args.smoke else cfg.get("attack", {}).get("budgets", [1, 2, 5, 10])
    surfaces = cfg.get("attack", {}).get("surfaces", ["ip", "nameserver", "registrar", "certificate"])
    manipulate_features = cfg.get("attack", {}).get("manipulate_features", True)

    print(f"{pfx}Evaluating structural attacks across budgets {budgets} over {len(seeds)} seed(s)...")

    results_by_budget: Dict[int, List[Dict[str, Any]]] = {k: [] for k in budgets}

    for s in seeds:
        run_name = f"attack_eval_seed_{s}"
        print(f"\n--- Seed {s} Attack Evaluation ---")

        if should_skip_run(paths.results_dir, run_name, force=args.force):
            print(f"  [RESUME] Skipping run: {run_name}")
            with open(paths.results_dir / "runs" / f"{run_name}.json", "r", encoding="utf-8") as f:
                r_cached = json.load(f)
            for k_str, b_stat in r_cached["budgets"].items():
                results_by_budget[int(k_str)].append(b_stat)
            continue

        splits = random_stratified_split(df["label"].to_numpy(), seed=s)
        clean_scaled, _ = apply_split_scaling(base_graph, splits.train_indices, val_indices=splits.val_indices)

        in_channels = {nt: clean_scaled[nt].x.shape[1] for nt in clean_scaled.node_types}

        # Model A: Standard SAGE (undefended)
        print(f"  Training clean SAGE model (Seed {s})...")
        model_sage = HeteroGNN(metadata=clean_scaled.metadata(), in_channels_dict=in_channels, variant="sage")
        eval_sage_clean, y_test, p_sage_clean, trained_sage = train_eval_gnn(
            model=model_sage, data=clean_scaled, splits=splits, df=df, epochs=train_epochs, patience=patience, seed=s
        )

        # Model D: SAGE_Guard (defense)
        print(f"  Training clean SAGE_Guard model (Seed {s})...")
        model_guard = HeteroGNN(metadata=clean_scaled.metadata(), in_channels_dict=in_channels, variant="sage_guard")
        eval_guard_clean, _, p_guard_clean, trained_guard = train_eval_gnn(
            model=model_guard, data=clean_scaled, splits=splits, df=df, epochs=train_epochs, patience=patience, seed=s
        )

        th_sage = eval_sage_clean["threshold"]
        th_01_sage = eval_sage_clean.get("operating_points", {}).get("tpr_at_0.1pct_fpr", {}).get("threshold", 0.5)

        th_guard = eval_guard_clean["threshold"]
        th_01_guard = eval_guard_clean.get("operating_points", {}).get("tpr_at_0.1pct_fpr", {}).get("threshold", 0.5)

        clean_mal_mask = (p_sage_clean[y_test == 1] >= th_sage)

        seed_budget_results = {}
        for k in budgets:
            print(f"    Evaluating attack budget k = {k}...")
            attacked_data = generate_structural_attack(
                clean_data=clean_scaled,
                splits=splits,
                budget=k,
                rho=1.0,
                surfaces=surfaces,
                manipulate_features=manipulate_features,
                seed=s
            )

            # (B) Attacked undefended SAGE
            res_b = evaluate_attacked_model(
                model=trained_sage,
                data=attacked_data,
                splits=splits,
                threshold=th_sage,
                operating_threshold_01=th_01_sage,
                clean_detected_mask=clean_mal_mask
            )

            # (C) Attacked defended SAGE_Guard
            res_c = evaluate_attacked_model(
                model=trained_guard,
                data=attacked_data,
                splits=splits,
                threshold=th_guard,
                operating_threshold_01=th_01_guard,
                clean_detected_mask=(p_guard_clean[y_test == 1] >= th_guard)
            )

            a_f1 = eval_sage_clean["f1"]
            b_f1 = res_b["f1"]
            c_f1 = res_c["f1"]
            d_f1 = eval_guard_clean["f1"]

            robustness_delta = b_f1 - a_f1
            denom_recovery = a_f1 - b_f1
            recovery = (c_f1 - b_f1) / denom_recovery if abs(denom_recovery) > 1e-6 else 1.0

            b_stat = {
                "budget": k,
                "clean_sage": {
                    "roc_auc": eval_sage_clean["roc_auc"],
                    "f1": a_f1,
                    "recall": eval_sage_clean["recall"],
                    "tpr_at_0.1pct_fpr": eval_sage_clean.get("operating_points", {}).get("tpr_at_0.1pct_fpr", {}).get("tpr", 0.0)
                },
                "attacked_sage_undefended": {
                    "roc_auc": res_b["roc_auc"],
                    "f1": b_f1,
                    "recall": res_b["recall"],
                    "tpr_at_0.1pct_fpr": res_b["tpr_at_0.1pct_fpr"],
                    "evasion_rate": res_b["evasion_rate"],
                    "robustness_delta": round(robustness_delta, 4)
                },
                "attacked_sage_guard_defended": {
                    "roc_auc": res_c["roc_auc"],
                    "f1": c_f1,
                    "recall": res_c["recall"],
                    "tpr_at_0.1pct_fpr": res_c["tpr_at_0.1pct_fpr"],
                    "evasion_rate": res_c["evasion_rate"],
                    "recovery_score": round(recovery, 4)
                },
                "clean_sage_guard_defense_cost": {
                    "f1": d_f1,
                    "f1_cost": round(d_f1 - a_f1, 4)
                }
            }
            results_by_budget[k].append(b_stat)
            seed_budget_results[str(k)] = b_stat

        save_run_result(paths.results_dir, run_name, {"experiment": "attack_eval", "seed": s, "budgets": seed_budget_results})

    # Aggregate over seeds
    aggregated_attack: Dict[str, Any] = {}
    for k, runs in results_by_budget.items():
        if not runs:
            continue
        clean_f1 = [r["clean_sage"]["f1"] for r in runs]
        atk_f1 = [r["attacked_sage_undefended"]["f1"] for r in runs]
        guard_f1 = [r["attacked_sage_guard_defended"]["f1"] for r in runs]
        evasion_rates = [r["attacked_sage_undefended"]["evasion_rate"] for r in runs]
        recoveries = [r["attacked_sage_guard_defended"]["recovery_score"] for r in runs]

        aggregated_attack[f"budget_{k}"] = {
            "budget": k,
            "num_seeds": len(runs),
            "clean_f1_mean": round(float(np.mean(clean_f1)), 4),
            "clean_f1_std": round(float(np.std(clean_f1)), 4),
            "attacked_undefended_f1_mean": round(float(np.mean(atk_f1)), 4),
            "attacked_undefended_f1_std": round(float(np.std(atk_f1)), 4),
            "attacked_guard_f1_mean": round(float(np.mean(guard_f1)), 4),
            "attacked_guard_f1_std": round(float(np.std(guard_f1)), 4),
            "evasion_rate_mean": round(float(np.mean(evasion_rates)), 4),
            "recovery_score_mean": round(float(np.mean(recoveries)), 4)
        }

    # Save JSON
    json_path = out_dir / "attack_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_attack, f, indent=2)

    # Markdown Summary Table
    pfx_t = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx_t}Adversarial Structural Attack & GNNGuard Recovery",
        "*Multi-instance PPT threat model: Attacker injects fake edges to top benign infrastructure*",
        f"*Evaluated across {len(seeds)} seeded runs (mean ± std)*",
        "",
        "| Budget (k) | (A) Clean SAGE F1 | (B) Attacked Undefended F1 | (C) Attacked + GNNGuard F1 | Evasion Rate | GNNGuard Recovery |",
        "|---|---|---|---|---|---|"
    ]
    for b_key, stat in aggregated_attack.items():
        k_val = stat["budget"]
        md_lines.append(
            f"| k = {k_val} | {stat['clean_f1_mean']:.4f} ± {stat['clean_f1_std']:.4f} | {stat['attacked_undefended_f1_mean']:.4f} ± {stat['attacked_undefended_f1_std']:.4f} | {stat['attacked_guard_f1_mean']:.4f} ± {stat['attacked_guard_f1_std']:.4f} | {stat['evasion_rate_mean']*100:.1f}% | {stat['recovery_score_mean']*100:.1f}% |"
        )
    md_lines.append("")
    md_lines.append("> **Metric O5 (Robustness)**: Recovery is computed as (C - B) / (A - B). Values >= 100% indicate full mitigation of injected adversarial edges.")

    md_path = out_dir / "attack_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== ATTACK EVALUATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
