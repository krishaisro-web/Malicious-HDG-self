#!/usr/bin/env python3
"""
tools/recover_pipeline_results.py
Post-hoc recovery utility for Malicious-HDG pipeline runs.

Recovers evaluation metrics, paired statistical comparisons (O3), latency benchmarks (O4),
and persists per-seed run payloads from EXISTING checkpoints WITHOUT retraining the GNN models.
Runtime: ~3-5 minutes on CPU (vs 4-5 days of full pipeline training).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    save_run_result
)
from src.hdg.data.graph import load_aligned_graph_cohort, apply_split_scaling
from src.hdg.data.splits import random_stratified_split
from src.hdg.cache import compute_experiment_fingerprint
from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.metrics import (
    evaluate_predictions,
    compute_bootstrap_cis,
    multi_seed_paired_comparisons
)
from src.hdg.training.baselines import run_xgboost_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover evaluation payloads and O3/O4 objectives from existing checkpoints.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke mode")
    parser.add_argument("--split-type", type=str, default="random", help="Split type to recover (default: random)")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    print("==========================================================================")
    print(" Malicious-HDG: Rapid Post-Hoc Pipeline Evaluation & Objective Recovery")
    print(f" Environment: {'FIXTURE' if args.fixture else ('SMOKE' if args.smoke else 'REAL')}")
    print(f" Results Dir: {paths.results_dir}")
    print(f" Checkpoints: {paths.checkpoints_dir}")
    print("==========================================================================")

    # 1. Load Aligned Dataset and Graph Cohort
    print("\n[Step 1/5] Loading aligned graph cohort and dataset...")
    base_graph, df, manifest = load_aligned_graph_cohort(paths.processed_dir)
    print(f"Loaded {len(df):,} domains and heterogeneous graph.")

    seeds = [42] if args.smoke else cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])
    variants = ["sage", "attn"]
    boot_samples = 100 if (args.smoke or args.fixture) else cfg.get("evaluation", {}).get("bootstrap_samples", 1000)

    # 2. Recompute / Ensure XGBoost Baselines with Persisted Predictions
    print(f"\n[Step 2/5] Verifying / Running XGBoost baselines across {len(seeds)} seeds...")
    xgb_predictions: Dict[int, Dict[str, np.ndarray]] = {}
    for s in seeds:
        r_name = f"baseline_xgboost_tabular_seed_{s}"
        r_path = paths.results_dir / "runs" / f"{r_name}.json"
        splits = random_stratified_split(df["label"].to_numpy(), seed=s)
        
        need_xgb = True
        if r_path.exists():
            try:
                with open(r_path, "r", encoding="utf-8") as f:
                    xgb_cached = json.load(f)
                if "predictions" in xgb_cached and "y_true" in xgb_cached["predictions"]:
                    need_xgb = False
                    p_true = np.array(xgb_cached["predictions"]["y_true"])
                    p_prob = np.array(xgb_cached["predictions"]["y_prob"])
                    th = float(xgb_cached.get("threshold", 0.5))
                    p_pred = np.array(xgb_cached["predictions"].get("y_pred", (p_prob >= th).astype(int)))
                    xgb_predictions[s] = {"y_true": p_true, "y_prob": p_prob, "y_pred": p_pred}
                    print(f"  [XGBoost Seed {s}] Loaded cached predictions (AUROC: {xgb_cached.get('roc_auc', 0.0):.4f})")
            except Exception:
                need_xgb = True

        if need_xgb:
            print(f"  [XGBoost Seed {s}] Computing baseline on random split...")
            res_xgb, y_test, test_probs = run_xgboost_baseline(df, splits, cfg, use_lexical=False, seed=s)
            y_pred = (test_probs >= res_xgb["threshold"]).astype(int)
            res_xgb["predictions"] = {
                "y_true": y_test.tolist(),
                "y_prob": test_probs.tolist(),
                "y_pred": y_pred.tolist()
            }
            save_run_result(paths.results_dir, r_name, res_xgb)
            xgb_predictions[s] = {"y_true": y_test, "y_prob": test_probs, "y_pred": y_pred}
            print(f"  [XGBoost Seed {s}] Computed & Saved (AUROC: {res_xgb['roc_auc']:.4f}, F1: {res_xgb['f1']:.4f})")

    # 3. Recover GNN Evaluation using Existing Saved Checkpoints
    print(f"\n[Step 3/5] Evaluating existing GNN checkpoints (ZERO RETRAINING)...")
    results_by_variant: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}
    predictions_by_variant: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {v: {} for v in variants}

    for var in variants:
        print(f"\n--- Architecture Variant: {var.upper()} ---")
        for s in seeds:
            ckpt_file = paths.checkpoints_dir / f"gnn_{var}_{args.split_type}_seed_{s}.pt"
            run_name = f"gnn_{var}_{args.split_type}_seed_{s}"
            
            if not ckpt_file.exists():
                print(f"  [WARN] Checkpoint not found: {ckpt_file.name}. Skipping seed {s}.")
                continue

            # Load checkpoint
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            ckpt_cfg = ckpt.get("model_config", {})
            th = float(ckpt.get("threshold", 0.5))

            # Split scaling
            splits = random_stratified_split(df["label"].to_numpy(), seed=s)
            scaled_data, _ = apply_split_scaling(
                base=base_graph,
                train_indices=splits.train_indices,
                val_indices=splits.val_indices,
                test_indices=splits.test_indices,
                degree_mode=cfg.get("graph", {}).get("degree_mode", "transductive")
            )

            in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
            model = HeteroGNN(
                metadata=scaled_data.metadata(),
                in_channels_dict=in_channels,
                variant=ckpt_cfg.get("variant", var),
                hidden_dim=ckpt_cfg.get("hidden_dim", 64),
                out_dim=ckpt_cfg.get("out_dim", 32),
                num_layers=ckpt_cfg.get("num_layers", 2),
                num_heads=ckpt_cfg.get("num_heads", 4),
                dropout=ckpt_cfg.get("dropout", 0.2),
                prune_threshold=ckpt_cfg.get("prune_threshold", 0.1),
                use_edges=ckpt_cfg.get("use_edges", True)
            )
            model.load_state_dict(state_dict)
            model.eval()

            # Forward pass on held-out test nodes
            with torch.no_grad():
                logits = model(scaled_data.x_dict, scaled_data.edge_index_dict)
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

            y_val_probs = probs[splits.val_indices]
            y_val_true = df["label"].to_numpy()[splits.val_indices]
            y_test_probs = probs[splits.test_indices]
            y_test_true = df["label"].to_numpy()[splits.test_indices]
            y_pred = (y_test_probs >= th).astype(int)

            eval_metrics = evaluate_predictions(
                y_true=y_test_true,
                y_prob=y_test_probs,
                threshold=th,
                y_val_true=y_val_true,
                y_val_prob=y_val_probs
            )
            eval_metrics["bootstrap_ci"] = compute_bootstrap_cis(
                y_true=y_test_true,
                y_prob=y_test_probs,
                threshold=th,
                n_bootstrap=boot_samples,
                seed=s
            )

            exp_fp = compute_experiment_fingerprint(
                cfg,
                stage=f"gnn_{var}_{args.split_type}",
                input_files=[paths.processed_dir / "heterodata.pt", paths.processed_dir / "domains.parquet"],
                extra_dict={"seed": s, "variant": var, "split_type": args.split_type}
            )

            run_payload = {
                "experiment": "gnn_evaluation",
                "status": "SUCCESS",
                "experiment_fingerprint": exp_fp,
                "variant": var,
                "split_type": args.split_type,
                "temporal_mode": None,
                "seed": s,
                "metrics": eval_metrics,
                "predictions": {
                    "y_true": y_test_true.tolist(),
                    "y_prob": y_test_probs.tolist(),
                    "y_pred": y_pred.tolist()
                }
            }
            save_run_result(paths.results_dir, run_name, run_payload)
            print(f"  [Seed {s}] Saved {run_name}.json (ROC-AUC: {eval_metrics['roc_auc']:.4f}, F1: {eval_metrics['f1']:.4f})")

            results_by_variant[var].append(eval_metrics)
            predictions_by_variant[var][s] = {
                "y_true": y_test_true,
                "y_prob": y_test_probs,
                "y_pred": y_pred
            }

    # 4. Compute Paired Hypothesis Tests (O3) and Aggregate Tables
    print(f"\n[Step 4/5] Computing paired hypothesis tests & statistical significance...")
    paired_results = {}
    if "sage" in variants and "attn" in variants and len(predictions_by_variant["sage"]) > 0:
        paired_results["sage_vs_attn"] = multi_seed_paired_comparisons(
            seeds=[s for s in seeds if s in predictions_by_variant["sage"] and s in predictions_by_variant["attn"]],
            predictions_a=predictions_by_variant["sage"],
            predictions_b=predictions_by_variant["attn"],
            n_bootstrap=boot_samples
        )

    if "sage" in variants and len(xgb_predictions) > 0:
        common_seeds = [s for s in seeds if s in predictions_by_variant["sage"] and s in xgb_predictions]
        paired_results["sage_vs_xgboost"] = multi_seed_paired_comparisons(
            seeds=common_seeds,
            predictions_a=predictions_by_variant["sage"],
            predictions_b=xgb_predictions,
            n_bootstrap=boot_samples
        )
        pooled = paired_results["sage_vs_xgboost"].get("pooled_summary", {})
        auc_diff = pooled.get("bootstrap_auc_diff", {})
        mcn = pooled.get("mcnemar", {})
        p_val_auc = auc_diff.get("pvalue_two_sided")
        mean_diff = auc_diff.get("mean_diff", 0.0)
        print(f"  ┌── [O3 STATISTICAL COMPARISON: HeteroGNN SAGE vs XGBoost] ─────────")
        print(f"  │  Mean Δ AUC: +{mean_diff:.4f} [95% CI: {auc_diff.get('ci_lower')}, {auc_diff.get('ci_upper')}]")
        print(f"  │  Paired Bootstrap AUC p-value: {p_val_auc}")
        print(f"  │  McNemar Test p-value:         {mcn.get('pvalue')}")
        print(f"  │  Statistical Significance:      {'SIGNIFICANT (p < 0.05) -> O3 MET' if (p_val_auc is not None and p_val_auc < 0.05 and mean_diff > 0) else 'NOT SIGNIFICANT'}")
        print(f"  └───────────────────────────────────────────────────────────────────")

    # Aggregate GNN stats
    aggregated_gnn = {}
    for var, runs in results_by_variant.items():
        if not runs:
            continue
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

    tables_dir = paths.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_payload = {
        "is_fixture": args.fixture,
        "is_smoke": args.smoke,
        "split_type": args.split_type,
        "variants": aggregated_gnn,
        "paired_comparisons": paired_results
    }
    with open(tables_dir / f"gnn_evaluation_{args.split_type}.json", "w", encoding="utf-8") as f:
        json.dump(table_payload, f, indent=2)

    # 5. Fast Latency Benchmark (O4)
    print(f"\n[Step 5/5] Benchmarking optimized CPU inference latency (O4)...")
    from src.hdg.eval.latency import benchmark_cpu_query_latency
    ref_ckpt = paths.checkpoints_dir / "gnn_sage_random_seed_42.pt"
    if ref_ckpt.exists():
        ckpt_obj = torch.load(ref_ckpt, map_location="cpu", weights_only=False)
        st_dict = ckpt_obj.get("model_state_dict", ckpt_obj)
        cfg_m = ckpt_obj.get("model_config", {})
        lat_model = HeteroGNN(
            metadata=scaled_data.metadata(),
            in_channels_dict=in_channels,
            variant=cfg_m.get("variant", "sage"),
            hidden_dim=cfg_m.get("hidden_dim", 64),
            out_dim=cfg_m.get("out_dim", 32),
            num_layers=cfg_m.get("num_layers", 2),
            num_heads=cfg_m.get("num_heads", 4),
            dropout=0.0,
            use_edges=cfg_m.get("use_edges", True)
        )
        lat_model.load_state_dict(st_dict)
    else:
        lat_model = HeteroGNN(metadata=scaled_data.metadata(), in_channels_dict=in_channels, variant="sage")

    n_q = 50 if (args.smoke or args.fixture) else 500
    w_up = 5 if (args.smoke or args.fixture) else 20
    lat_stats = benchmark_cpu_query_latency(
        model=lat_model,
        data=scaled_data,
        df=df,
        batch_size=1,
        num_queries=n_q,
        warmup=w_up,
        seed=42
    )
    e2e = lat_stats["end_to_end"]
    print(f"  CPU Batch-1 Latency: Mean = {e2e['mean_ms']:.2f} ms, P95 = {e2e['p95_ms']:.2f} ms")
    print(f"  Budget Target (< 50 ms): {'MET' if (e2e['mean_ms'] < 50.0 and e2e['p95_ms'] < 50.0) else 'NOT MET'}")

    # Update latency_results.json
    lat_path = tables_dir / "latency_results.json"
    if lat_path.exists():
        try:
            with open(lat_path, "r", encoding="utf-8") as f:
                lat_json = json.load(f)
            # update batch 1 benchmark
            for b in lat_json.get("benchmarks", []):
                if b.get("batch_size") == 1:
                    b.update(lat_stats)
            with open(lat_path, "w", encoding="utf-8") as f:
                json.dump(lat_json, f, indent=2)
        except Exception:
            pass

    # Compile Final Report
    print(f"\nCompiling final report with updated O1-O5 objectives...")
    report_script = REPO_ROOT / "pipeline" / "11_make_report.py"
    import subprocess
    cmd = [sys.executable, str(report_script), "--force"]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.fixture:
        cmd.append("--fixture")
    if args.smoke:
        cmd.append("--smoke")
    subprocess.run(cmd, check=True)

    print("\n==========================================================================")
    print(" RECOVERY COMPLETE! Report regenerated at: results/real/REPORT.md")
    print("==========================================================================")


if __name__ == "__main__":
    main()
