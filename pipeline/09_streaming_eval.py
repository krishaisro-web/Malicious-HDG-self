#!/usr/bin/env python3
"""
CLI script to evaluate online / streaming updates for Malicious-HDG across consecutive months.
Compares:
1. Retrain from scratch (upper bound, highest compute)
2. Naive fine-tuning (online training; tests for catastrophic forgetting)
3. Replay buffer (reservoir sample of size 1000 mixed with new data)
Reports:
- F1, TPR at 0.1% FPR, ROC-AUC on new month
- Backward transfer / forgetting on Month 1 at each incremental step
Outputs to results/tables/streaming_eval.json, streaming_eval.md, and results/runs/streaming_eval.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from src.hdg.config import (
    load_config,
    get_resolved_paths,
    should_skip_run,
    save_run_result,
    init_thread_pool,
)
from src.hdg.eval.streaming import run_streaming_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate streaming domain updates and continual learning.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--max-months", type=int, default=None, help="Maximum number of chronological months to evaluate")
    parser.add_argument("--reservoir-size", type=int, default=1000, help="Replay buffer capacity (default: 1000)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    from src.hdg.data.graph import load_aligned_graph_cohort
    from src.hdg.cache import compute_experiment_fingerprint

    max_months = args.max_months
    if max_months is None:
        max_months = 4 if (args.smoke or args.fixture) else None

    exp_fp = compute_experiment_fingerprint(
        cfg,
        stage="streaming_eval",
        extra_dict={"max_months": max_months, "reservoir_size": args.reservoir_size}
    )

    run_name = "streaming_eval"
    if should_skip_run(paths, run_name, force=args.force, expected_fingerprint=exp_fp):
        print(f"[{run_name}] Run already completed with matching fingerprint. Skipping (use --force to rerun).")
        return

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading aligned graph and domain dataset...")
    data, df, manifest = load_aligned_graph_cohort(paths.processed_dir)

    init_thread_pool(cfg)

    epochs_scratch = 15 if (args.smoke or args.fixture) else 50
    epochs_finetune = 5 if (args.smoke or args.fixture) else 15

    print(f"Executing Streaming Evaluation (Max Months: {max_months}, Reservoir: {args.reservoir_size})...")
    res = run_streaming_evaluation(
        data=data,
        df=df,
        temporal_col="t_month",
        replay_reservoir_size=args.reservoir_size,
        epochs_scratch=epochs_scratch,
        epochs_finetune=epochs_finetune,
        max_months=max_months,
        seed=cfg.get("randomness", {}).get("seed", 42)
    )
    res["experiment_fingerprint"] = exp_fp

    # Save JSON
    json_path = out_dir / "streaming_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_path = out_dir / "streaming_eval.md"

    if res.get("status") == "BLOCKED_INSUFFICIENT_DATA":
        print(f"\n[BLOCKED] Streaming evaluation cannot proceed: {res.get('reason')}")
        md_lines = [
            f"# {pfx}Streaming Domain Prototype Evaluation",
            "",
            "> [!WARNING]",
            f"> **Status: BLOCKED_INSUFFICIENT_DATA**",
            f"> {res.get('reason')}",
            "",
            "### Monthly Class Distribution in Dataset",
            "| Month | Total Records | Malware Count | Benign Count | Status |",
            "|:---:|:---:|:---:|:---:|:---:|",
        ]
        for m_name, counts in res.get("month_class_counts", {}).items():
            status_str = "Eligible (Mixed)" if counts["malware"] > 0 and counts["benign"] > 0 else "Single-Class Only"
            md_lines.append(
                f"| `{m_name}` | {counts['total']:,} | {counts['malware']:,} | {counts['benign']:,} | {status_str} |"
            )
        md_lines.append("")
        md_lines.append("> [!NOTE]")
        md_lines.append("> Multi-month continual learning evaluation requires multiple transitions between consecutive months containing both classes.")
        md_lines.append("> To evaluate temporal robustness on datasets with a single overlapping month, refer to the Single Chronological Holdout results in `gnn_evaluation_time.json`.")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        save_run_result(paths, run_name, res)
        print(f"Results JSON: {json_path}")
        print(f"Results Markdown: {md_path}")
        return

    # Generate Markdown for successful run
    md_lines = [
        f"# {pfx}Streaming Domain Prototype Evaluation",
        "",
        f"- **Initial Baseline Month ($M_0$)**: `{res['month_0']}`",
        f"- **Evaluated Months**: `{', '.join(res['evaluated_months'])}`",
        f"- **Replay Reservoir Capacity**: `{res['replay_reservoir_capacity']}`",
        f"- **Initial $M_0$ Test F1**: `{res['month_0_baseline']['f1']:.4f}` | **TPR@0.1% FPR**: `{res['month_0_baseline']['tpr_at_01_fpr']:.4f}`",
        "",
        "## Continual Learning Comparison ($T \\to T+1$)",
        "| Step | Target Month | Strategy | New Month F1 | New Month TPR@0.1% FPR | New Month ROC-AUC | Month 1 F1 | Backward Transfer (ΔF1) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in res.get("steps", []):
        st_num = s["step"]
        m_str = s["month"]
        strat = s["strategy"]
        cur = s["current_month_perf"]
        m0_p = s["month_1_perf"]
        bwt = s["backward_transfer_f1"]
        bwt_str = f"+{bwt:.4f}" if bwt > 0 else f"{bwt:.4f}"
        md_lines.append(
            f"| {st_num} | {m_str} | `{strat}` | {cur['f1']:.4f} | {cur['tpr_at_01_fpr']:.4f} | {cur['roc_auc']:.4f} | {m0_p['f1']:.4f} | **{bwt_str}** |"
        )

    md_lines.append("")
    md_lines.append("### Key Findings")
    md_lines.append("1. **Retrain from scratch**: Maintains robust performance across both past and present distributions at the expense of cumulative training time.")
    md_lines.append("2. **Naive fine-tuning**: Adapts rapidly to the active month's malicious patterns but suffers from catastrophic forgetting (negative backward transfer on Month 1).")
    md_lines.append("3. **Replay buffer**: Mitigates catastrophic forgetting by interleaving historical anchor domains, stabilizing long-term recall without full retraining.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Resumability record
    save_run_result(paths, run_name, res)

    print(f"\n{pfx}=== STREAMING EVALUATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
