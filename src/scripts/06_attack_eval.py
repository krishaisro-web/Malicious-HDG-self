#!/usr/bin/env python3
"""
CLI script to evaluate adversarial robustness of Malicious-HDG models
under structural edge-injection attacks (budgets k in {1, 2, 5}).
Outputs to results/tables/attack_results.json and attack_results.md.
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

from models.hetero_gnn import HeteroGNN
from src.hdg.attack import run_structural_attack
from src.hdg.config import load_config, get_resolved_paths
from src.hdg.splits import random_stratified_split
from src.hdg.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Adversarial Structural Attack.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError("Processed files not found. Run 01_parse.py and 02_build_graph.py first.")

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading graph and domains...")
    df = pd.read_parquet(parquet_file)
    data = torch.load(graph_file, weights_only=False)

    splits = random_stratified_split(df["label"].to_numpy(), seed=args.seed)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Training clean baseline model...")
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage"
    )

    _, _, _, trained_model = train_eval_gnn(
        model=model,
        data=data,
        splits=splits,
        df=df,
        epochs=args.epochs,
        seed=args.seed
    )

    budgets = cfg.get("evaluation", {}).get("attack_budgets", [1, 2, 5])
    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Running structural attack across budgets: {budgets}...")

    attack_results = run_structural_attack(
        model=trained_model,
        clean_data=data,
        splits=splits,
        budgets=budgets,
        seed=args.seed
    )

    # Save JSON
    json_path = out_dir / "attack_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(attack_results, f, indent=2)

    # Save Markdown
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}Adversarial Structural Attack Evaluation",
        "*Attacker injects edges from malicious test domains to popular benign infrastructure*",
        "",
        f"**Clean Test ROC-AUC**: {attack_results['clean_metrics']['roc_auc']:.4f}",
        f"**Clean Test Recall**: {attack_results['clean_metrics']['recall']:.4f}",
        "",
        "| Edge Budget (k) | Attacked ROC-AUC | AUC Drop | Evasion Count | Evasion Rate |",
        "|---|---|---|---|---|"
    ]
    for b_key, b_stat in attack_results.get("budgets", {}).items():
        k_val = b_key.replace("budget_", "")
        md_lines.append(
            f"| k = {k_val} | {b_stat['roc_auc']:.4f} | {b_stat['auc_drop']:+.4f} | {b_stat['evaded_count']} | {b_stat['evasion_rate']*100:.1f}% |"
        )

    md_lines.append("")
    md_lines.append("> **Note**: GNNGuard and graph defense mechanisms are identified as future work.")

    md_path = out_dir / "attack_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== ATTACK EVALUATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
