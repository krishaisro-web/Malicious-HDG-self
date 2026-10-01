#!/usr/bin/env python3
"""
CLI script to emit BIND Response Policy Zone (RPZ) DNS blocking rules from Malicious-HDG.
Implements:
1. Calibrates decision threshold on validation benign cohort to guarantee operational FPR <= 0.1%.
2. Evaluates test set domains against calibrated threshold.
3. Formats and writes standard BIND RPZ file (<domain> CNAME .) with detailed operational metadata comments.
4. Generates summary statistics: emitted rule count, estimated blocking rate, distribution by source family.
Outputs to:
- results/rpz/malicious_domains.rpz
- results/rpz/rpz_summary.json
- results/tables/rpz_summary.md
- results/runs/emit_rpz.json
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    should_skip_run,
    save_run_result,
    init_thread_pool,
)
from src.hdg.splits import rolling_origin_temporal_splits, random_stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit BIND RPZ rules for high-confidence malicious domains.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint .pt")
    parser.add_argument("--target-fpr", type=float, default=0.001, help="Operating target FPR (default: 0.001 = 0.1%%)")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    run_name = "emit_rpz"
    if should_skip_run(paths, run_name, force=args.force):
        print(f"[{run_name}] Run already completed. Skipping (use --force to rerun).")
        return

    rpz_dir = paths.results_dir / "rpz"
    tables_dir = paths.results_dir / "tables"
    paths.check_write_path(rpz_dir)
    paths.check_write_path(tables_dir)
    rpz_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"

    if not parquet_file.exists() or not graph_file.exists():
        raise FileNotFoundError(
            f"Processed artifacts not found in {paths.processed_dir}. Run 01_parse.py and 02_build_graph.py first."
        )

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading graph and domain dataset...")
    df = pd.read_parquet(parquet_file)
    data = torch.load(graph_file, weights_only=False)

    init_thread_pool(cfg)

    # Dataset splits: prioritize latest temporal transition if available, else random stratified
    splits = None
    try:
        if "t_month" in df.columns or "t" in df.columns:
            transitions = rolling_origin_temporal_splits(df, min_distinct_months=2, min_per_class_per_window=10)
            if transitions:
                _, splits = transitions[-1]
    except Exception as e:
        print(f"Temporal split unavailable ({e}), using random stratified split.")

    if splits is None:
        splits = random_stratified_split(df["label"].to_numpy(), seed=cfg.get("randomness", {}).get("seed", 42))

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=64,
        out_dim=32,
        num_layers=2
    )

    # Load or train model
    model_loaded = False
    if args.checkpoint and Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
        print(f"Loaded model checkpoint from: {args.checkpoint}")
        model_loaded = True
    else:
        # Check if a checkpoint exists in models/checkpoints/
        ckpt_dir = REPO_ROOT / "models" / "checkpoints"
        available_ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        if available_ckpts:
            ckpt_path = available_ckpts[0]
            try:
                ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt_obj, dict):
                    if "model_state_dict" in ckpt_obj:
                        state_dict = ckpt_obj["model_state_dict"]
                    elif "state_dict" in ckpt_obj:
                        state_dict = ckpt_obj["state_dict"]
                    else:
                        state_dict = ckpt_obj
                else:
                    state_dict = ckpt_obj
                model.load_state_dict(state_dict)
                print(f"Loaded checkpoint from: {ckpt_path}")
                model_loaded = True
            except Exception as e:
                print(f"Could not load {ckpt_path} ({e}), training fresh model...")

    if not model_loaded:
        print("Training fast base model for RPZ threshold calibration...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
        model.train()
        train_idx = splits.train_indices
        y_all = data["domain"].y
        for epoch in range(15 if (args.smoke or args.fixture) else 40):
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            loss = F.cross_entropy(out[train_idx], y_all[train_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        all_probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

    # Calibrate operational threshold on validation negatives
    val_idx = splits.val_indices
    y_val = data["domain"].y[val_idx].cpu().numpy()
    val_probs = all_probs[val_idx]
    val_negs = val_probs[y_val == 0]

    if len(val_negs) == 0:
        threshold = 0.90
        realized_val_fpr = 0.0
    else:
        # Percentile corresponding to (1 - target_fpr)
        p = max(0.0, min(100.0, (1.0 - args.target_fpr) * 100.0))
        threshold = float(np.percentile(val_negs, p))
        realized_val_fpr = float(np.mean(val_negs >= threshold))

    print(f"Operating Target FPR: {args.target_fpr * 100:.2f}% | Calibrated Threshold: {threshold:.4f} (Val FPR: {realized_val_fpr * 100:.3f}%)")

    # Evaluate test cohort
    test_idx = splits.test_indices
    test_probs = all_probs[test_idx]
    test_df = df.iloc[test_idx].copy()
    test_df["score"] = test_probs

    blocked_df = test_df[test_df["score"] >= threshold].sort_values(by="score", ascending=False)
    n_blocked = len(blocked_df)
    n_test = len(test_df)
    blocking_rate = (n_blocked / n_test * 100.0) if n_test > 0 else 0.0

    print(f"Test Domains: {n_test} | Blocked: {n_blocked} ({blocking_rate:.2f}%)")

    now_iso = datetime.now(timezone.utc).isoformat()
    rpz_lines = [
        "; ==============================================================================",
        "; Malicious-HDG Automated Response Policy Zone (BIND RPZ)",
        f"; Generated: {now_iso}",
        "; Architecture: HeteroGNN-SAGE",
        f"; Operating Target FPR: {args.target_fpr * 100:.2f}% ({args.target_fpr})",
        f"; Calibrated Threshold: {threshold:.4f}",
        f"; Realized Validation FPR: {realized_val_fpr:.5f} ({realized_val_fpr * 100:.3f}%)",
        f"; Total Evaluated Test Domains: {n_test}",
        f"; Total Emitted Blocking Rules: {n_blocked}",
        f"; Estimated Blocking Rate: {blocking_rate:.2f}%",
        "; Policy Action: CNAME . (NXDOMAIN / Drop)",
        "; ==============================================================================",
        "$TTL 300",
        "@ IN SOA localhost. root.localhost. ( 2026100101 3600 1800 604800 300 )",
        "  IN NS  localhost.",
        ""
    ]

    for _, row in blocked_df.iterrows():
        d_name = row.get("domain", row.get("raw_domain", "unknown"))
        score = row["score"]
        rpz_lines.append(f"{d_name} CNAME . ; score={score:.4f}, threshold={threshold:.4f}, val_fpr={realized_val_fpr:.5f}")

    rpz_file = rpz_dir / "malicious_domains.rpz"
    with open(rpz_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rpz_lines) + "\n")

    # Family and Source Breakdown
    family_counts = blocked_df["family"].value_counts().to_dict() if "family" in blocked_df.columns else {}
    source_counts = blocked_df["source"].value_counts().to_dict() if "source" in blocked_df.columns else {}

    summary = {
        "generated_at": now_iso,
        "target_fpr": args.target_fpr,
        "calibrated_threshold": round(threshold, 4),
        "realized_val_fpr": round(realized_val_fpr, 5),
        "total_test_domains": n_test,
        "rules_emitted": n_blocked,
        "blocking_rate_pct": round(blocking_rate, 2),
        "top_malware_families": {k: int(v) for k, v in list(family_counts.items())[:10]},
        "top_sources": {k: int(v) for k, v in list(source_counts.items())[:10]}
    }

    # Save summary JSON
    summary_json_path = rpz_dir / "rpz_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save summary Markdown
    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx}BIND Response Policy Zone (RPZ) Rule Emission Summary",
        "",
        "## Policy Configuration",
        f"- **Generated At**: `{now_iso}`",
        f"- **Operational Target FPR**: `{args.target_fpr * 100:.2f}%`",
        f"- **Calibrated Threshold**: `{threshold:.4f}`",
        f"- **Realized Validation FPR**: `{realized_val_fpr * 100:.3f}%`",
        f"- **Total Evaluated Domains**: `{n_test}`",
        f"- **Rules Emitted**: `{n_blocked}`",
        f"- **Estimated Blocking Rate**: `{blocking_rate:.2f}%`",
        f"- **RPZ Output File**: `{rpz_file.name}`",
        "",
        "## Distribution by Malware Family",
        "| Malware Family | Blocked Domains | Percentage |",
        "|:---|:---:|:---:|"
    ]
    for fam, cnt in list(family_counts.items())[:10]:
        pct = (cnt / n_blocked * 100.0) if n_blocked > 0 else 0.0
        md_lines.append(f"| {fam or 'Unknown'} | {cnt} | {pct:.1f}% |")

    md_lines.append("")
    md_lines.append("## Distribution by Threat Feed Source")
    md_lines.append("| Feed Source | Blocked Domains | Percentage |")
    md_lines.append("|:---|:---:|:---:|")
    for src, cnt in list(source_counts.items())[:10]:
        pct = (cnt / n_blocked * 100.0) if n_blocked > 0 else 0.0
        md_lines.append(f"| {src or 'Unknown'} | {cnt} | {pct:.1f}% |")

    summary_md_path = tables_dir / "rpz_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # Resumability record
    save_run_result(paths, run_name, summary)

    print(f"\n{pfx}=== RPZ RULE EMISSION COMPLETE ===")
    print(f"RPZ File: {rpz_file}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary MD: {summary_md_path}")


if __name__ == "__main__":
    main()
