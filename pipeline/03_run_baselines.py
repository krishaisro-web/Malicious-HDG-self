#!/usr/bin/env python3
"""
CLI script to run baseline models on identical splits for Malicious-HDG.
Evaluates across 5 seeds:
- TF-IDF (char 2-5) + Logistic Regression on e2LD
- XGBoost on Tabular non-lexical
- XGBoost on Tabular + Lexical
- Single-feature length baseline
- Single-feature no_ip baseline
- Isolated Domain MLP (HeteroGNN with use_edges=False)
Supports --smoke, --max-domains N, and --force for resumability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.hdg.training.baselines import (
    run_tfidf_lexical_baseline,
    run_xgboost_baseline,
    run_single_feature_baseline,
    run_mlp_no_edges_baseline,
)
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    should_skip_run,
    save_run_result
)
from src.hdg.data.graph import apply_split_scaling
from src.hdg.data.splits import random_stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baselines on identical splits.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed only")
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
    if not parquet_file.exists():
        raise FileNotFoundError(f"Domains file not found: {parquet_file}")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    max_d = args.max_domains or (5000 if args.smoke else cfg.get("graph", {}).get("max_domains"))
    if max_d is not None and len(df) > max_d:
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    base_graph = torch.load(graph_file, weights_only=False) if graph_file.exists() else None

    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.smoke:
        seeds = [42]
    else:
        seeds = cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])

    print(f"{pfx}Running baselines across {len(seeds)} seed(s): {seeds}...")

    models = [
        "tfidf_logreg_e2ld",
        "xgboost_tabular",
        "xgboost_tabular_lexical",
        "single_feat_length",
        "single_feat_no_ip",
        "mlp_no_edges"
    ]
    results_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}

    for s in seeds:
        print(f"\n--- Evaluating Seed {s} ---")
        splits = random_stratified_split(df["label"].to_numpy(), seed=s)

        # 1. TF-IDF + Logistic Regression
        r_name = f"baseline_tfidf_logreg_seed_{s}"
        if not should_skip_run(paths.results_dir, r_name, force=args.force):
            res_tfidf, _, _ = run_tfidf_lexical_baseline(df, splits, cfg, seed=s)
            save_run_result(paths.results_dir, r_name, res_tfidf)
        else:
            with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                res_tfidf = json.load(f)
        results_by_model["tfidf_logreg_e2ld"].append(res_tfidf)
        print(f"  [TF-IDF LogReg] ROC-AUC: {res_tfidf['roc_auc']:.4f}, F1: {res_tfidf['f1']:.4f}")

        # 2. XGBoost Tabular
        r_name = f"baseline_xgboost_tabular_seed_{s}"
        if not should_skip_run(paths.results_dir, r_name, force=args.force):
            res_xgb_tab, _, _ = run_xgboost_baseline(df, splits, cfg, use_lexical=False, seed=s)
            save_run_result(paths.results_dir, r_name, res_xgb_tab)
        else:
            with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                res_xgb_tab = json.load(f)
        results_by_model["xgboost_tabular"].append(res_xgb_tab)
        print(f"  [XGBoost Tabular] ROC-AUC: {res_xgb_tab['roc_auc']:.4f}, F1: {res_xgb_tab['f1']:.4f}")

        # 3. XGBoost Tabular + Lexical
        r_name = f"baseline_xgboost_lexical_seed_{s}"
        if not should_skip_run(paths.results_dir, r_name, force=args.force):
            res_xgb_all, _, _ = run_xgboost_baseline(df, splits, cfg, use_lexical=True, seed=s)
            save_run_result(paths.results_dir, r_name, res_xgb_all)
        else:
            with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                res_xgb_all = json.load(f)
        results_by_model["xgboost_tabular_lexical"].append(res_xgb_all)
        print(f"  [XGBoost Tabular+Lexical] ROC-AUC: {res_xgb_all['roc_auc']:.4f}, F1: {res_xgb_all['f1']:.4f}")

        # 4. Single-feature length
        r_name = f"baseline_single_length_seed_{s}"
        if not should_skip_run(paths.results_dir, r_name, force=args.force):
            res_len, _, _ = run_single_feature_baseline(df, splits, feature_name="length", seed=s)
            save_run_result(paths.results_dir, r_name, res_len)
        else:
            with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                res_len = json.load(f)
        results_by_model["single_feat_length"].append(res_len)
        print(f"  [Single-Feat Length] ROC-AUC: {res_len['roc_auc']:.4f}, F1: {res_len['f1']:.4f}")

        # 5. Single-feature no_ip
        r_name = f"baseline_single_no_ip_seed_{s}"
        if not should_skip_run(paths.results_dir, r_name, force=args.force):
            res_noip, _, _ = run_single_feature_baseline(df, splits, feature_name="no_ip", seed=s)
            save_run_result(paths.results_dir, r_name, res_noip)
        else:
            with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                res_noip = json.load(f)
        results_by_model["single_feat_no_ip"].append(res_noip)
        print(f"  [Single-Feat No-IP] ROC-AUC: {res_noip['roc_auc']:.4f}, F1: {res_noip['f1']:.4f}")

        # 6. Isolated Domain MLP (HeteroGNN with use_edges=False)
        if base_graph is not None:
            r_name = f"baseline_mlp_no_edges_seed_{s}"
            if not should_skip_run(paths.results_dir, r_name, force=args.force):
                scaled_g, _ = apply_split_scaling(base_graph, splits.train_indices)
                epochs_mlp = 3 if (args.smoke or args.fixture) else 30
                res_mlp, _, _ = run_mlp_no_edges_baseline(scaled_g, splits, df, cfg, epochs=epochs_mlp, seed=s)
                save_run_result(paths.results_dir, r_name, res_mlp)
            else:
                with open(paths.results_dir / "runs" / f"{r_name}.json", "r", encoding="utf-8") as f:
                    res_mlp = json.load(f)
            results_by_model["mlp_no_edges"].append(res_mlp)
            print(f"  [Isolated Domain MLP] ROC-AUC: {res_mlp['roc_auc']:.4f}, F1: {res_mlp['f1']:.4f}")

        import gc
        gc.collect()

    # Aggregate mean +/- std across seeds
    aggregated: Dict[str, Any] = {}
    for m_name, runs in results_by_model.items():
        if not runs:
            continue
        aucs = [r["roc_auc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        prs = [r["pr_auc"] for r in runs]
        aggregated[m_name] = {
            "num_seeds": len(runs),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_std": round(float(np.std(prs)), 4),
            "runs": runs
        }

    # Save JSON
    json_path = out_dir / "baselines_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Markdown Summary
    pfx_title = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx_title}Baseline Model Evaluation (Identical Splits)",
        f"*Evaluated across {len(seeds)} seeded splits (mean ± std)*",
        "",
        "| Model | ROC-AUC | F1 Score | PR-AUC |",
        "|---|---|---|---|"
    ]
    for m_name, stat in aggregated.items():
        md_lines.append(
            f"| `{m_name}` | {stat['roc_auc_mean']:.4f} ± {stat['roc_auc_std']:.4f} | {stat['f1_mean']:.4f} ± {stat['f1_std']:.4f} | {stat['pr_auc_mean']:.4f} ± {stat['pr_auc_std']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.")

    md_path = out_dir / "baselines_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== BASELINES COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
