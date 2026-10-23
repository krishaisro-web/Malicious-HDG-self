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
REPO_ROOT = Path(__file__).resolve().parent.parent
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
from src.hdg.data.splits import rolling_origin_temporal_splits, random_stratified_split


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

    from src.hdg.data.graph import load_aligned_graph_cohort

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Loading aligned graph and domain dataset...")
    data, df, manifest = load_aligned_graph_cohort(paths.processed_dir)

    init_thread_pool(cfg)

    # Dataset splits: prioritize latest temporal transition if available, else random stratified
    splits = None
    try:
        if "t_month" in df.columns or "t" in df.columns:
            transitions = rolling_origin_temporal_splits(df, min_distinct_months=2, min_per_class_per_window=10, allow_empty=True)
            if transitions:
                _, splits = transitions[-1]
    except Exception as e:
        print(f"Temporal split unavailable ({e}), using random stratified split.")

    if splits is None:
        splits = random_stratified_split(df["label"].to_numpy(), seed=cfg.get("randomness", {}).get("seed", 42))

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    # Resolve validated model checkpoint
    model = None
    model_loaded = False
    resolved_ckpt_path = None
    ckpt_manifest = {}

    candidates_to_try = []
    if args.checkpoint:
        cand = Path(args.checkpoint)
        if cand.exists():
            candidates_to_try.append(cand)
    else:
        ckpt_dir = paths.checkpoints_dir
        preferred_candidates = [
            ckpt_dir / "gnn_attn_random_seed_42.pt",
            ckpt_dir / "gnn_sage_guard_random_seed_42.pt",
            ckpt_dir / "gnn_sage_random_seed_42.pt"
        ]
        for p in preferred_candidates:
            if p.exists() and p not in candidates_to_try:
                candidates_to_try.append(p)
        if ckpt_dir.exists():
            for p in sorted(ckpt_dir.glob("*.pt")):
                if not p.name.endswith("_train_state.pt") and p not in candidates_to_try:
                    candidates_to_try.append(p)

    for cand_path in candidates_to_try:
        try:
            ckpt_obj = torch.load(cand_path, map_location="cpu", weights_only=False)
            state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj

            # Check if checkpoint manifest exists
            man_path = cand_path.parent / f"{cand_path.stem}_manifest.json"
            cand_manifest = {}
            if man_path.exists():
                with open(man_path, "r", encoding="utf-8") as f:
                    cand_manifest = json.load(f)

            # Strict model_config validation (fail-closed, no silent defaults or guessing)
            ckpt_cfg = ckpt_obj.get("model_config") if isinstance(ckpt_obj, dict) else None
            if not isinstance(ckpt_cfg, dict):
                raise ValueError("Checkpoint is missing required 'model_config' dictionary.")

            required_fields = ["variant", "hidden_dim", "out_dim", "num_layers", "num_heads", "dropout", "prune_threshold", "use_edges"]
            missing_fields = [f for f in required_fields if f not in ckpt_cfg]
            if missing_fields:
                raise ValueError(f"Checkpoint model_config is incomplete; missing required fields: {missing_fields}")

            variant = str(ckpt_cfg["variant"]).lower()
            if variant not in ["sage", "attn", "sage_guard"]:
                raise ValueError(f"Unsupported architecture variant: '{variant}'. Must be one of ['sage', 'attn', 'sage_guard']")

            hidden_dim = int(ckpt_cfg["hidden_dim"])
            out_dim = int(ckpt_cfg["out_dim"])
            num_layers = int(ckpt_cfg["num_layers"])
            num_heads = int(ckpt_cfg["num_heads"])
            dropout = float(ckpt_cfg["dropout"])
            prune_threshold = float(ckpt_cfg["prune_threshold"])
            use_edges = bool(ckpt_cfg["use_edges"])

            # Dynamically instantiate matching architecture strictly with validated parameters
            cand_model = HeteroGNN(
                metadata=data.metadata(),
                in_channels_dict=in_channels,
                variant=variant,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                prune_threshold=prune_threshold,
                use_edges=use_edges
            )
            cand_model.load_state_dict(state_dict)
            model = cand_model
            resolved_ckpt_path = cand_path
            ckpt_manifest = cand_manifest
            model_loaded = True
            print(f"[RPZ] Successfully loaded verified model checkpoint: {resolved_ckpt_path.name} "
                  f"(variant={variant}, hidden_dim={hidden_dim}, out_dim={out_dim}, layers={num_layers}, heads={num_heads})")
            if ckpt_manifest:
                print(f"[RPZ] Verified checkpoint manifest: {man_path.name}")
            break
        except Exception as e:
            print(f"[WARN] Candidate checkpoint {cand_path.name} could not be loaded ({e}), trying next candidate...")

    if not model_loaded or model is None:
        print("[BLOCKED] RPZ rule generation blocked: No verified production model checkpoint found.")
        print("          Run 'pipeline/04_train_gnn.py' first to train, evaluate, and save calibrated checkpoints.")
        blocked_summary = {
            "status": "BLOCKED_MISSING_MODEL",
            "reason": f"No verified model checkpoint found in {paths.checkpoints_dir}. Will not emit RPZ rules from unrecorded model.",
            "checkpoint_path": None,
            "rules_emitted": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        # Fail-closed: remove any stale RPZ zone file
        rpz_file = rpz_dir / "malicious_domains.rpz"
        if rpz_file.exists():
            rpz_file.unlink()
        summary_json_path = rpz_dir / "rpz_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(blocked_summary, f, indent=2)
        save_run_result(paths, run_name, blocked_summary)
        return

    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        all_probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

    # Split validation negatives into calibration cohort (for setting threshold)
    # and independent certification holdout (for verifying FPR without data reuse)
    val_idx = splits.val_indices
    y_val = data["domain"].y[val_idx].cpu().numpy()
    val_probs = all_probs[val_idx]
    val_negs = val_probs[y_val == 0]

    n_val_negs = len(val_negs)
    if n_val_negs < 2:
        print("[BLOCKED] RPZ rule generation blocked: Fewer than 2 validation negatives available.")
        blocked_summary = {
            "status": "BLOCKED_INSUFFICIENT_VALIDATION_NEGS",
            "reason": "Calibration requires observed benign validation instances to bound False Positive Rate. Cannot certify zero FPR.",
            "n_val_negs": n_val_negs,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        rpz_file = rpz_dir / "malicious_domains.rpz"
        if rpz_file.exists():
            rpz_file.unlink()
        with open(rpz_dir / "rpz_summary.json", "w", encoding="utf-8") as f:
            json.dump(blocked_summary, f, indent=2)
        save_run_result(paths, run_name, blocked_summary)
        return

    # Two-fold split of validation negatives: 50% for order-statistic threshold calibration,
    # 50% independent holdout for rigorous statistical certification (prevents optimistic bias from data reuse)
    rng = np.random.RandomState(cfg.get("randomness", {}).get("seed", 42))
    shuffled_indices = rng.permutation(n_val_negs)
    n_calib = max(1, n_val_negs // 2)
    calib_negs = val_negs[shuffled_indices[:n_calib]]
    cert_negs = val_negs[shuffled_indices[n_calib:]]
    n_cert = len(cert_negs)

    # Statistical Certification Requirement:
    # Under exact binomial model, for 0 false positives on independent verification holdout:
    # 1 - 0.05^(1/N_cert) <= target_fpr  ==>  N_cert >= log(0.05) / log(1 - target_fpr)
    min_cert_negs = int(np.ceil(np.log(0.05) / np.log(1.0 - args.target_fpr))) if args.target_fpr < 1.0 else 1

    # Order-statistic threshold calibration on calibration cohort
    sorted_calib = np.sort(calib_negs)
    max_calib_fp = int(np.floor(len(calib_negs) * args.target_fpr))
    rank_idx = len(calib_negs) - 1 - max_calib_fp
    threshold = float(sorted_calib[-1]) + 1e-6 if rank_idx < 0 else float(sorted_calib[rank_idx])

    # Now certify threshold independently on certification holdout (cert_negs)
    observed_fp_cert = int(np.sum(cert_negs >= threshold))
    realized_cert_fpr = float(observed_fp_cert / n_cert)

    # Compute exact 95% confidence upper bound on population FPR using independent certification holdout
    if observed_fp_cert == 0:
        fpr_95_upper_bound = float(1.0 - 0.05 ** (1.0 / n_cert))
        confidence_bound_type = "Exact Binomial (1 - 0.05^(1/N_cert))"
    else:
        z = 1.96
        p_hat = realized_cert_fpr
        denom = 1.0 + (z**2) / n_cert
        centre = p_hat + (z**2) / (2.0 * n_cert)
        spread = z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * n_cert)) / n_cert)
        fpr_95_upper_bound = min(1.0, float((centre + spread) / denom))
        confidence_bound_type = "Wilson Score (95% Two-Sided Upper)"

    # Strict Certification Gate:
    # Fail closed if sample size is below required certification power OR if the 95% upper bound exceeds target FPR
    cert_passed = (n_cert >= min_cert_negs) and (fpr_95_upper_bound <= args.target_fpr)
    if not cert_passed:
        print(f"[BLOCKED] RPZ rule generation blocked: Statistical certification failed. "
              f"95% Confidence Upper Bound on FPR ({fpr_95_upper_bound * 100:.3f}%) exceeds target FPR ({args.target_fpr * 100:.3f}%) "
              f"or certification negatives (N={n_cert}) below required minimum ({min_cert_negs}).")
        blocked_summary = {
            "status": "BLOCKED_UNCERTIFIED_FPR",
            "reason": f"95% confidence upper bound ({fpr_95_upper_bound * 100:.3f}%) exceeds operational target FPR ({args.target_fpr * 100:.3f}%).",
            "target_fpr": args.target_fpr,
            "calibrated_threshold": round(threshold, 4),
            "n_calibration_negs": len(calib_negs),
            "n_certification_negs": n_cert,
            "min_required_certification_negs": min_cert_negs,
            "observed_fps_on_certification": observed_fp_cert,
            "realized_cert_fpr": round(realized_cert_fpr, 5),
            "fpr_95_upper_bound": round(fpr_95_upper_bound, 5),
            "confidence_bound_type": confidence_bound_type,
            "rules_emitted": 0,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        rpz_file = rpz_dir / "malicious_domains.rpz"
        if rpz_file.exists():
            rpz_file.unlink()
        with open(rpz_dir / "rpz_summary.json", "w", encoding="utf-8") as f:
            json.dump(blocked_summary, f, indent=2)
        save_run_result(paths, run_name, blocked_summary)
        return

    print(f"Operational Target FPR: {args.target_fpr * 100:.3f}% | Calibrated Threshold: {threshold:.4f} "
          f"(Certified FPR Upper Bound: {fpr_95_upper_bound * 100:.3f}% <= {args.target_fpr * 100:.3f}%, "
          f"Observed FP on Holdout: {observed_fp_cert}/{n_cert})")

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
    arch_label = f"HeteroGNN-{getattr(model, 'variant', 'unknown').upper()}"
    rpz_lines = [
        "; ==============================================================================",
        "; Malicious-HDG Automated Response Policy Zone (BIND RPZ)",
        f"; Generated: {now_iso}",
        f"; Architecture: {arch_label}",
        f"; Checkpoint File: {resolved_ckpt_path.name if resolved_ckpt_path else 'None'}",
        f"; Operating Target FPR: {args.target_fpr * 100:.3f}% ({args.target_fpr})",
        f"; Calibrated Threshold: {threshold:.4f}",
        f"; Realized Certification FPR: {realized_cert_fpr:.5f} ({realized_cert_fpr * 100:.3f}%)",
        f"; Independent Holdout Negatives: {n_cert}",
        f"; False Positives on Holdout: {observed_fp_cert}",
        f"; 95% Confidence Upper Bound on FPR: {fpr_95_upper_bound * 100:.3f}% ({confidence_bound_type})",
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
        rpz_lines.append(f"{d_name} CNAME . ; score={score:.4f}, threshold={threshold:.4f}, cert_fpr={realized_cert_fpr:.5f}")

    rpz_file = rpz_dir / "malicious_domains.rpz"
    with open(rpz_file, "w", encoding="utf-8") as f:
        f.write("\n".join(rpz_lines) + "\n")

    # Family and Source Breakdown
    family_counts = blocked_df["family"].value_counts().to_dict() if "family" in blocked_df.columns else {}
    source_counts = blocked_df["source"].value_counts().to_dict() if "source" in blocked_df.columns else {}

    summary = {
        "generated_at": now_iso,
        "checkpoint_file": resolved_ckpt_path.name if resolved_ckpt_path else None,
        "architecture": getattr(model, "variant", "unknown"),
        "target_fpr": args.target_fpr,
        "calibrated_threshold": round(threshold, 4),
        "calibration_negatives": len(calib_negs),
        "independent_certification_negatives": n_cert,
        "observed_false_positives_holdout": observed_fp_cert,
        "realized_cert_fpr": round(realized_cert_fpr, 5),
        "fpr_95_upper_bound": round(fpr_95_upper_bound, 5),
        "confidence_bound_type": confidence_bound_type,
        "certified_within_target": True,
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
        f"- **Model Checkpoint**: `{resolved_ckpt_path.name if resolved_ckpt_path else 'None'}` ({getattr(model, 'variant', 'unknown')})",
        f"- **Operational Target FPR**: `{args.target_fpr * 100:.2f}%`",
        f"- **Calibrated Threshold**: `{threshold:.4f}`",
        f"- **Validation Negatives Evaluated**: `{n_val_negs}`",
        f"- **Observed False Positives**: `{observed_fp}`",
        f"- **Realized Validation FPR**: `{realized_val_fpr * 100:.3f}%`",
        f"- **95% Confidence Upper Bound on Population FPR**: `{fpr_95_upper_bound * 100:.2f}%`",
        f"- **Statistical Evidence Certification**: `{'CERTIFIED' if adequate_evidence else 'EMPIRICAL (Limited Sample Size)'}`",
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
