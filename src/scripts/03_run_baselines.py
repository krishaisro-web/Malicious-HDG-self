#!/usr/bin/env python3
"""
CLI script to run baseline models on identical splits for Malicious-HDG.
Evaluates:
- TF-IDF (char 2-5) + Logistic Regression on e2LD
- XGBoost on Tabular non-lexical
- XGBoost on Tabular + Lexical
- Single-feature length baseline
Saves baselines_results.json and baselines_results.md.
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

from src.hdg.baselines import run_tfidf_lexical_baseline, run_xgboost_baseline
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.splits import random_stratified_split, group_split_bipartite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Baselines on identical splits.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed only")
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
    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Running baselines across {len(seeds)} seed(s): {seeds}...")

    results_by_model: Dict[str, List[Dict[str, Any]]] = {
        "tfidf_logreg_e2ld": [],
        "xgboost_tabular": [],
        "xgboost_tabular_lexical": []
    }

    for s in seeds:
        print(f"\n--- Evaluating Seed {s} ---")
        splits = random_stratified_split(df["label"].to_numpy(), seed=s)

        # 1. TF-IDF + Logistic Regression
        res_tfidf, _, _ = run_tfidf_lexical_baseline(df, splits, cfg, seed=s)
        results_by_model["tfidf_logreg_e2ld"].append(res_tfidf)
        print(f"  [TF-IDF LogReg] ROC-AUC: {res_tfidf['roc_auc']:.4f}, F1: {res_tfidf['f1']:.4f}")

        # 2. XGBoost Tabular (Non-lexical)
        res_xgb_tab, _, _ = run_xgboost_baseline(df, splits, cfg, use_lexical=False, seed=s)
        results_by_model["xgboost_tabular"].append(res_xgb_tab)
        print(f"  [XGBoost Tabular] ROC-AUC: {res_xgb_tab['roc_auc']:.4f}, F1: {res_xgb_tab['f1']:.4f}")

        # 3. XGBoost Tabular + Lexical
        res_xgb_all, _, _ = run_xgboost_baseline(df, splits, cfg, use_lexical=True, seed=s)
        results_by_model["xgboost_tabular_lexical"].append(res_xgb_all)
        print(f"  [XGBoost Tabular+Lexical] ROC-AUC: {res_xgb_all['roc_auc']:.4f}, F1: {res_xgb_all['f1']:.4f}")

    # Aggregate mean +/- std across seeds (never reporting only a best seed)
    aggregated: Dict[str, Any] = {}
    for m_name, runs in results_by_model.items():
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
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}Baseline Model Evaluation (Identical Splits)",
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
