#!/usr/bin/env python3
"""
CLI script to evaluate Two-Stage Hybrid Embedding Stacker (GNN + XGBoost) and
CVaR Top-K pAUC Fine-Tuning against pre-committed stop rules (RFC-002 Pillars 4 & 5).
Stop Rule (docs/analysis_plan.md Condition 6):
  Delta TPR @ 0.1% FPR >= +3.0 percentage points with 95% paired bootstrap CI > 0.
  Otherwise, recorded transparently as a pre-committed negative result.
Saves results to results/.../tables/stacker_results.json and pauc_finetune_results.json.
"""

import argparse
from copy import deepcopy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.hdg.config import load_config, get_resolved_paths, init_thread_pool
from src.hdg.data.features import get_primary_domain_node_type
from src.hdg.data.graph import apply_split_scaling, load_aligned_graph_cohort
from src.hdg.data.splits import random_stratified_split, stratified_apex_cluster_split
from src.hdg.device import resolve_device, setup_shared_gpu_environment
from src.hdg.metrics import apply_temperature_scaling, fit_temperature_scaling
from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.training.stacker import EmbeddingXGBStacker, evaluate_stacker_against_stop_rule
from src.hdg.training.train import fine_tune_pauc_gnn


def _resolve_checkpoint(paths: Any, split_type: str, explicit_ckpt: Optional[str]) -> Optional[Path]:
    if explicit_ckpt:
        p = Path(explicit_ckpt)
        return p if p.exists() else None
    search_dirs = [
        REPO_ROOT / "artifacts" / "checkpoints" / "real_gpu",
        paths.checkpoints_dir,
        REPO_ROOT / "artifacts" / "checkpoints" / "real",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for name in (
            f"gnn_attn_{split_type}_seed_42.pt",
            f"gnn_sage_{split_type}_seed_42.pt",
            "gnn_attn_random_seed_42.pt",
            "gnn_sage_random_seed_42.pt",
        ):
            cand = d / name
            if cand.exists():
                return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stop rules for Stacker and pAUC fine-tuning.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'cuda', 'cuda:N')")
    parser.add_argument("--split-type", type=str, default="random", choices=["random", "apex_cluster"], help="Evaluation split")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit GNN checkpoint path")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save output tables")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    device = resolve_device(args.device)
    if device.type == "cuda":
        mem_frac = float(os.environ.get("HDG_GPU_MEMORY_FRACTION", "0.95"))
        min_vram = float(os.environ.get("MIN_FREE_VRAM", "10.0"))
        setup_shared_gpu_environment(
            dev_idx=(device.index if device.index is not None else 0),
            memory_fraction=mem_frac,
            min_free_vram_gb=min_vram,
        )

    if args.out_dir:
        tables_dir = Path(args.out_dir)
        if "real_gpu" in str(tables_dir) or device.type == "cuda":
            paths.checkpoints_dir = REPO_ROOT / "artifacts" / "checkpoints" / "real_gpu"
    else:
        tables_dir = paths.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pauc_out_file = tables_dir / "pauc_finetune_results.json"
    stacker_out_file = tables_dir / "stacker_results.json"

    print("[STOP RULES] Evaluating Hybrid Embedding Stacker & Chunked CVaR pAUC Fine-Tuning (RFC-002 Pillars 4-5)...")

    processed_exists = (paths.processed_dir / "heterodata.pt").exists()
    ckpt_path = _resolve_checkpoint(paths, args.split_type, args.checkpoint)

    if not processed_exists or ckpt_path is None:
        reason = "processed graph missing" if not processed_exists else "no saved GNN seed-42 checkpoint found"
        print(f"  [BLOCKED] Cannot execute live Stage 6d evaluation ({reason}).")
        blocked: Dict[str, Any] = {
            "status": "BLOCKED_MISSING_PREREQUISITES",
            "is_positive_result": False,
            "reason": reason,
            "stop_rule_threshold": 0.030,
        }
        with open(stacker_out_file, "w", encoding="utf-8") as f:
            json.dump(blocked, f, indent=2)
        with open(pauc_out_file, "w", encoding="utf-8") as f:
            json.dump(blocked, f, indent=2)
        return

    base_graph, df, _ = load_aligned_graph_cohort(paths.processed_dir)
    seed = 42
    if args.split_type == "apex_cluster":
        splits = stratified_apex_cluster_split(df, seed=seed)
    else:
        splits = random_stratified_split(df["label"].to_numpy(), seed=seed)

    deg_mode = cfg.get("graph", {}).get("degree_mode", "transductive")
    max_hub_fanout = cfg.get("graph", {}).get("max_hub_fanout")
    scaled_data, _ = apply_split_scaling(
        base=base_graph,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
        degree_mode=deg_mode,
        mask_disjoint_groups=(args.split_type == "apex_cluster"),
        max_hub_fanout=max_hub_fanout,
    )

    ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_cfg = ckpt_obj.get("model_config", {})
    in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
    base_model = HeteroGNN(
        metadata=scaled_data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=int(m_cfg.get("hidden_dim", 64)),
        out_dim=int(m_cfg.get("out_dim", 32)),
        num_layers=int(m_cfg.get("num_layers", 2)),
        num_heads=int(m_cfg.get("num_heads", 4)),
        dropout=float(m_cfg.get("dropout", 0.2)),
        variant=str(m_cfg.get("variant", "sage")),
        prune_threshold=float(m_cfg.get("prune_threshold", 0.1)),
        use_edges=bool(m_cfg.get("use_edges", True)),
        use_gated_agg=bool(m_cfg.get("use_gated_agg", True)),
        use_char_cnn=bool(m_cfg.get("use_char_cnn", False)),
        char_cnn_dim=int(m_cfg.get("char_cnn_dim", 16)),
        use_completeness_gate=bool(m_cfg.get("use_completeness_gate", False)),
        num_envs=int(m_cfg.get("num_envs", 0)),
    )
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj)
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)
    scaled_dev = scaled_data.to(device)

    p_nt = get_primary_domain_node_type(scaled_dev)
    ew_dict = {
        et: scaled_dev[et].edge_weight
        for et in scaled_dev.edge_types
        if hasattr(scaled_dev[et], "edge_weight") and scaled_dev[et].edge_weight is not None
    } or None
    char_bytes = getattr(scaled_dev[p_nt], "char_bytes", None)
    y_all = scaled_dev[p_nt].y.cpu().numpy().astype(int)
    dom_x = scaled_dev[p_nt].x.cpu().numpy()

    base_model.eval()
    with torch.no_grad():
        base_logits, penult_emb = base_model.get_penultimate_embeddings(
            scaled_dev.x_dict,
            scaled_dev.edge_index_dict,
            edge_weight_dict=ew_dict,
            char_bytes=char_bytes,
        )
        base_logits_np = base_logits.cpu().numpy()
        emb_np = penult_emb.cpu().numpy()

    y_val = y_all[splits.val_indices]
    y_test = y_all[splits.test_indices]
    tr_cfg = cfg.get("model", {}).get("training", {})
    use_temp = bool(tr_cfg.get("use_temperature_scaling", True))
    temp = fit_temperature_scaling(base_logits_np[splits.val_indices], y_val) if use_temp else 1.0
    base_val_probs = apply_temperature_scaling(base_logits_np[splits.val_indices], temperature=temp)
    base_test_probs = apply_temperature_scaling(base_logits_np[splits.test_indices], temperature=temp)

    # 1. Hybrid Representation Stacker: [z_i || x_i] -> XGBoost
    stacker = EmbeddingXGBStacker(
        max_depth=4,
        subsample=0.8,
        n_estimators=100,
        learning_rate=0.1,
        random_state=seed,
    )
    fit_meta = stacker.fit(
        embeddings=emb_np,
        domain_features=dom_x,
        y_all=y_all,
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        threshold_criterion=str(tr_cfg.get("threshold_criterion", "f0.5")),
        max_val_fpr=float(tr_cfg.get("max_validation_fpr", 0.015)),
    )
    stack_metrics, stack_val_probs, stack_test_probs = stacker.evaluate(
        embeddings=emb_np,
        domain_features=dom_x,
        y_all=y_all,
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
    )
    stacker_stop = evaluate_stacker_against_stop_rule(
        y_test=y_test,
        sage_probs=base_test_probs,
        stacker_probs=stack_test_probs,
        y_val=y_val,
        sage_val_probs=base_val_probs,
        stacker_val_probs=stack_val_probs,
        stop_rule_gain_threshold=0.030,
        seed=seed,
    )
    stacker_result = {
        "status": stacker_stop["status"],
        "is_positive_result": stacker_stop["is_positive_result"],
        "checkpoint": str(ckpt_path),
        "base_sage_tpr_at_01pct_fpr": stacker_stop["tpr_sage_at_01pct_fpr"],
        "intervention_tpr_at_01pct_fpr": stacker_stop["tpr_stacker_at_01pct_fpr"],
        "delta_tpr": stacker_stop["delta_tpr"],
        "bootstrap_ci_95": list(stacker_stop["bootstrap_ci_95"]),
        "stop_rule_threshold": 0.030,
        "stacker_metrics": stack_metrics,
        "stacker_fit_metadata": {k: v for k, v in fit_meta.items() if k != "val_probs"},
        "interpretation": stacker_stop["interpretation"],
    }
    with open(stacker_out_file, "w", encoding="utf-8") as f:
        json.dump(stacker_result, f, indent=2)
    print(f"  [STOP RULE] Hybrid Stacker -> {stacker_result['status']} (F1={stack_metrics['f1']:.4f}, Δ@0.1%FPR={stacker_stop['delta_tpr']:+.4f})")

    # 2. Chunked CVaR Top-K pAUC Fine-Tuning
    ft_epochs = 5 if (args.smoke or args.fixture) else 30
    ft_model = deepcopy(base_model).to(device)
    ft_eval, _, ft_test_probs, _ = fine_tune_pauc_gnn(
        model=ft_model,
        data=scaled_data,
        splits=splits,
        df=df,
        epochs=ft_epochs,
        lr=float(tr_cfg.get("pauc_lr", 0.0005)),
        patience=10,
        k_fraction=float(tr_cfg.get("pauc_k_fraction", 0.01)),
        margin=float(tr_cfg.get("pauc_margin", 0.20)),
        lambda_pauc=float(tr_cfg.get("pauc_lambda", 0.50)),
        max_val_fpr=0.01,
        seed=seed,
        device=device,
        loss_type=str(tr_cfg.get("loss_type", "focal")),
        focal_alpha=float(tr_cfg.get("focal_alpha", 0.75)),
        focal_gamma=float(tr_cfg.get("focal_gamma", 2.0)),
        threshold_criterion=str(tr_cfg.get("threshold_criterion", "f0.5")),
        max_validation_fpr=float(tr_cfg.get("max_validation_fpr", 0.015)),
        use_temperature_scaling=use_temp,
    )
    ft_val_probs = ft_eval.pop("val_probs", base_val_probs)
    pauc_stop = evaluate_stacker_against_stop_rule(
        y_test=y_test,
        sage_probs=base_test_probs,
        stacker_probs=ft_test_probs,
        y_val=y_val,
        sage_val_probs=base_val_probs,
        stacker_val_probs=ft_val_probs,
        stop_rule_gain_threshold=0.030,
        seed=seed,
    )
    pauc_result = {
        "status": pauc_stop["status"],
        "is_positive_result": pauc_stop["is_positive_result"],
        "checkpoint": str(ckpt_path),
        "base_sage_tpr_at_01pct_fpr": pauc_stop["tpr_sage_at_01pct_fpr"],
        "intervention_tpr_at_01pct_fpr": pauc_stop["tpr_stacker_at_01pct_fpr"],
        "delta_tpr": pauc_stop["delta_tpr"],
        "bootstrap_ci_95": list(pauc_stop["bootstrap_ci_95"]),
        "stop_rule_threshold": 0.030,
        "finetune_metrics": ft_eval,
        "interpretation": pauc_stop["interpretation"],
    }
    with open(pauc_out_file, "w", encoding="utf-8") as f:
        json.dump(pauc_result, f, indent=2)
    print(f"  [STOP RULE] pAUC Fine-Tune -> {pauc_result['status']} (F1={ft_eval['f1']:.4f}, Δ@0.1%FPR={pauc_stop['delta_tpr']:+.4f})")


if __name__ == "__main__":
    main()
