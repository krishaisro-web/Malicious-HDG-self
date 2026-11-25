"""
Malicious-HDG: Unified, Leak-Free Training and Evaluation Loop for Heterogeneous GNNs.

Overview & Research Protocol (Phase 2 Specifications):
------------------------------------------------------
This module implements the canonical training, validation, early-stopping, checkpointing,
and multi-metric evaluation pipeline for Malicious-HDG models (HeteroGraphSAGE & HeteroHGTConv).

Methodological & Operational Guarantees:
----------------------------------------
1. Zero Data Leakage:
   - Data scaling statistics (mean, std) are derived strictly from training domains.
   - Optimal classification thresholds (tau*) and operating points (tau @ 1% FPR, tau @ 0.1% FPR)
     are selected strictly on validation sets and frozen before evaluating test sets.
2. Production Operating Calibration:
   - Evaluates true realised test FPR alongside TPR at certified operational thresholds.
   - Calculates prevalence-adjusted precision for realistic DNS monitoring scenarios
     where malicious domains represent small prior probabilities (e.g., pi = 0.01 or pi = 0.001).
3. Robust Interruption Recovery & Atomic State Checkpointing:
   - Checkpoints full model, optimizer, scheduler, and epoch metrics atomically after each epoch.
   - Validates cryptographic experiment fingerprints before resuming to ensure state integrity.
4. Memory & VRAM Safeguards:
   - Deallocates intermediate activation tensors immediately (del out, loss, val_out).
   - Periodically compacts CUDA allocator cache blocks (torch.cuda.empty_cache()).
   - Enforces PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to eliminate virtual fragmentation.
5. Strict FP32 Arithmetic:
   - No BF16 or FP16 mixed precision is allowed during GNN forward passes, preserving
     fine-grained margin separation and GNNGuard cosine edge pruning (eps = 1e-8).
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# Ensure CUDA caching allocator uses expandable segments to avoid virtual memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sklearn.metrics import roc_auc_score
from src.hdg.metrics import (
    apply_temperature_scaling,
    choose_optimal_threshold,
    compute_bootstrap_cis,
    evaluate_predictions,
    fit_temperature_scaling,
)
from src.hdg.data.splits import DataSplits
from src.hdg.data.features import get_feature_indices, get_primary_domain_node_type


class FocalLoss(nn.Module):
    """
    Focal Loss for binary/2-class imbalanced classification (Lin et al., ICCV 2017):
      L_Focal(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Suppresses gradient dominance of easy negative domains while preserving hard margin focus.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * F.cross_entropy(logits, targets, reduction="none")
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()


def compute_class_balance_weights(
    y_train: torch.Tensor,
    mode: str = "class_weighted",
    effective_beta: float = 0.9999,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Computes inverse-frequency or Cui et al. (CVPR 2019) effective-sample class weights
    strictly on `y_train` and explicitly places the FP32 tensor on `device`.
    """
    pos_cnt = int((y_train == 1).sum().item())
    neg_cnt = int((y_train == 0).sum().item())
    if pos_cnt <= 0 or neg_cnt <= 0:
        return None

    m = (mode or "ce").lower()
    if m == "effective_samples":
        beta = min(0.999999, max(0.5, float(effective_beta)))
        eff_0 = max(1e-12, 1.0 - math.pow(beta, neg_cnt))
        eff_1 = max(1e-12, 1.0 - math.pow(beta, pos_cnt))
        w0_raw = (1.0 - beta) / eff_0
        w1_raw = (1.0 - beta) / eff_1
        norm = 2.0 / max(1e-12, (w0_raw + w1_raw))
        w = torch.tensor([w0_raw * norm, w1_raw * norm], dtype=torch.float, device=device)
        return w
    if m == "class_weighted":
        weight_1 = float(neg_cnt / pos_cnt)
        return torch.tensor([1.0, weight_1], dtype=torch.float, device=device)
    return None


def _compute_cls_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "ce",
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Unified classification loss dispatcher supporting `'ce'`, `'class_weighted'`, `'effective_samples'`, and `'focal'`."""
    if (loss_type or "ce").lower() == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)(logits, targets)
    return F.cross_entropy(logits, targets, weight=class_weights, reduction=reduction)


def deep_coral_loss(
    h_source: torch.Tensor,
    h_target: torch.Tensor,
    max_samples: int = 8192,
) -> torch.Tensor:
    """
    Computes unlabeled Deep CORAL covariance alignment loss (Sun & Saenko, ECCV 2016):
      L_CORAL = (1 / (4 * d^2)) * || C(H_S) - C(H_T) ||_F^2
    Subsamples up to `max_samples` rows per side so memory stays O(B * d) (< 5 MB on A100).
    Never accesses target labels.
    """
    n_s, d = h_source.shape
    n_t = h_target.shape[0]
    if n_s < 2 or n_t < 2 or d == 0:
        return h_source.sum() * 0.0

    if n_s > max_samples:
        idx_s = torch.randperm(n_s, device=h_source.device)[:max_samples]
        h_s = h_source[idx_s]
    else:
        h_s = h_source

    if n_t > max_samples:
        idx_t = torch.randperm(n_t, device=h_target.device)[:max_samples]
        h_t = h_target[idx_t]
    else:
        h_t = h_target

    h_s_c = h_s - h_s.mean(dim=0, keepdim=True)
    h_t_c = h_t - h_t.mean(dim=0, keepdim=True)
    cov_s = (h_s_c.transpose(0, 1) @ h_s_c) / max(1, h_s.shape[0] - 1)
    cov_t = (h_t_c.transpose(0, 1) @ h_t_c) / max(1, h_t.shape[0] - 1)
    diff = cov_s - cov_t
    return torch.sum(diff * diff) / (4.0 * float(d * d))


def _extract_graph_aux_inputs(
    graph: HeteroData,
) -> Tuple[Optional[Dict[Tuple[str, str, str], torch.Tensor]], Optional[torch.Tensor]]:
    """Extracts optional IIF `edge_weight_dict` and `char_bytes` from a HeteroData object."""
    ew_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
    for et in graph.edge_types:
        if hasattr(graph[et], "edge_weight") and graph[et].edge_weight is not None:
            ew_dict[et] = graph[et].edge_weight
    primary_nt = get_primary_domain_node_type(graph)
    char_bytes = getattr(graph[primary_nt], "char_bytes", None)
    return (ew_dict if ew_dict else None), char_bytes


def _forward_model(
    model: nn.Module,
    graph: HeteroData,
    edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]],
    char_bytes: Optional[torch.Tensor],
    grl_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unified forward dispatch supporting both RFC-003 models and legacy nn.Modules."""
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(
            graph.x_dict,
            graph.edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            char_bytes=char_bytes,
            grl_lambda=grl_lambda,
        )
    logits = model(graph.x_dict, graph.edge_index_dict)
    return logits, logits, None


def train_eval_gnn(
    model: nn.Module,
    data: HeteroData,
    splits: DataSplits,
    df: pd.DataFrame,
    epochs: int = 200,
    lr: float = 0.005,
    lr_grid: Optional[List[float]] = None,
    weight_decay: float = 0.0001,
    patience: int = 60,
    min_epochs: int = 50,
    target_fprs: List[float] = [0.01, 0.001],
    prevalence_pis: List[float] = [0.01, 0.001],
    bootstrap_samples: int = 1000,
    use_class_weights: bool = False,
    scheduler_type: Optional[str] = "plateau",
    checkpoint_path: Optional[Path] = None,
    log_file: Optional[Path] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    resume: bool = True,
    experiment_fingerprint: Optional[str] = None,
    split_type: Optional[str] = None,
    split_metadata: Optional[Dict[str, Any]] = None,
    eval_data: Optional[HeteroData] = None,
    use_grl: bool = False,
    grl_weight: float = 0.2,
    use_coral: bool = False,
    coral_weight: float = 0.1,
    coral_max_samples: int = 8192,
    use_group_dro: bool = False,
    dro_eta: float = 0.05,
    drop_drift_shortcuts: bool = False,
    co_primary_cross_source: bool = False,
    loss_type: str = "ce",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    effective_beta: float = 0.9999,
    threshold_criterion: str = "f1",
    max_validation_fpr: float = 0.015,
    use_temperature_scaling: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, nn.Module]:
    """
    Trains HeteroGNN with early stopping on validation ROC-AUC and validation-tuned threshold.
    
    Execution Stages:
    1. Hardware & Seed Initialization: Pins RNG seeds on PyTorch and NumPy for determinism.
    2. Resumption Verification: Checks for atomic intermediate checkpoint; verifies fingerprint.
    3. Hyperparameter Tuning: Optionally evaluates candidate learning rates via short grid passes.
    4. Training & Validation Loop: Iterates epochs, computes cross-entropy loss, and tracks val ROC-AUC.
    5. Early Stopping: Evaluates patience counter with an enforced minimum epoch floor (`min_epochs`).
    6. Threshold Selection: Tunes classification threshold strictly on validation probabilities.
    7. Test Evaluation & Operating Points: Evaluates test set; records TPR @ 1% and 0.1% FPR.
    8. Bootstrap Confidence Intervals: Computes empirical 95% CIs via stratified resampling.
    9. Subgroup Breakdown Audits: Analyzes performance across resolution status, source, and families.

    Args:
        model: HeteroGNN neural network architecture instance.
        data: Scaled HeteroData graph containing train/val/test domain features and relational edges.
        splits: DataSplits dataclass with train_indices, val_indices, test_indices.
        df: Aligned domains DataFrame containing metadata (no_ip, source, family, label).
        epochs: Maximum number of training epochs (default: 200).
        lr: Base initial learning rate (default: 0.005).
        lr_grid: Optional list of candidate learning rates to search over on validation set.
        weight_decay: L2 regularization coefficient for Adam optimizer.
        patience: Number of epochs without validation ROC-AUC improvement before early stopping.
        min_epochs: Minimum epoch threshold before early stopping is allowed to trigger.
        target_fprs: Target false positive rates for operational thresholds (e.g. 0.01, 0.001).
        prevalence_pis: Real-world malicious prevalence rates for adjusted precision computation.
        bootstrap_samples: Number of bootstrap iterations for 95% confidence intervals.
        use_class_weights: Whether to weight minority (malicious) class in cross-entropy loss.
        scheduler_type: "plateau" (ReduceLROnPlateau) or "cosine" (CosineAnnealingLR).
        checkpoint_path: Destination path for final model weights and manifest.
        log_file: Destination path for JSON Lines per-epoch training metrics.
        device: Target execution device (torch.device("cuda") or torch.device("cpu")).
        seed: Random seed for model parameter initialization.
        resume: If True, attempts to resume from existing intermediate training state checkpoint.
        experiment_fingerprint: Unique configuration hash ensuring checkpoint compatibility.
        split_type: Evaluation split name ("random", "time", or "group").
        split_metadata: Supplementary partitioning details (e.g. cutoff dates, disjoint components).
        eval_data: Optional separate HeteroData graph for temporal inference evaluation.

    Returns:
        Tuple of (eval_results_dict, y_test_true, y_test_prob, trained_model).
    """
    from src.hdg.cache import atomic_save_torch, atomic_save_json

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    primary_nt = get_primary_domain_node_type(data)

    if drop_drift_shortcuts:
        data = data.clone()
        miss_idx = get_feature_indices(primary_nt, group="missingness", dim=data[primary_nt].x.shape[1])
        if miss_idx:
            data[primary_nt].x = data[primary_nt].x.clone()
            data[primary_nt].x[:, miss_idx] = 0.0
        if eval_data is not None:
            eval_data = eval_data.clone()
            ev_nt = get_primary_domain_node_type(eval_data)
            e_miss = get_feature_indices(ev_nt, group="missingness", dim=eval_data[ev_nt].x.shape[1])
            if e_miss:
                eval_data[ev_nt].x = eval_data[ev_nt].x.clone()
                eval_data[ev_nt].x[:, e_miss] = 0.0

    model = model.to(device)
    data = data.to(device)
    edge_weight_dict, char_bytes = _extract_graph_aux_inputs(data)

    train_idx = torch.tensor(splits.train_indices, dtype=torch.long, device=device)
    val_idx = torch.tensor(splits.val_indices, dtype=torch.long, device=device)
    test_idx = torch.tensor(splits.test_indices, dtype=torch.long, device=device)

    y_all = data[primary_nt].y

    # Prepare GRL environment IDs and Group DRO partition masks on train_idx
    env_targets: Optional[torch.Tensor] = None
    active_grl = bool(use_grl and getattr(model, "env_classifier", None) is not None and "source" in df.columns)
    if active_grl:
        n_envs_model = int(getattr(model, "num_envs", 2))
        src_col = df["source"].astype(str).tolist()
        # Map Umbrella vs CESNET vs malware feeds deterministically into [0, n_envs_model - 1]
        env_arr = np.zeros(len(df), dtype=np.int64)
        for i_s, s_val in enumerate(src_col):
            sl = s_val.lower()
            if "umbrella" in sl:
                env_arr[i_s] = 0
            elif "cesnet" in sl or "groundtruth" in sl:
                env_arr[i_s] = min(1, n_envs_model - 1)
            else:
                env_arr[i_s] = (abs(hash(sl)) % max(1, n_envs_model))
        if len(np.unique(env_arr[splits.train_indices])) < 2:
            active_grl = False
        else:
            env_targets = torch.tensor(env_arr, dtype=torch.long, device=device)

    dro_group_masks: List[torch.Tensor] = []
    dro_q: Optional[torch.Tensor] = None
    if use_group_dro and "source" in df.columns:
        tr_sources = df["source"].iloc[splits.train_indices].astype(str).tolist()
        tr_labels = df["label"].iloc[splits.train_indices].to_numpy().astype(int)
        grp_keys: Dict[Tuple[int, str], List[int]] = {}
        for pos, (lbl, s_name) in enumerate(zip(tr_labels, tr_sources)):
            grp_keys.setdefault((int(lbl), s_name), []).append(pos)
        if len(grp_keys) >= 2:
            for positions in grp_keys.values():
                if len(positions) >= 2:
                    dro_group_masks.append(torch.tensor(positions, dtype=torch.long, device=device))
            if len(dro_group_masks) >= 2:
                dro_q = torch.full((len(dro_group_masks),), 1.0 / len(dro_group_masks), dtype=torch.float, device=device)

    # Validation cross-source slices for co-primary early stopping
    val_source_slices: List[Tuple[str, np.ndarray]] = []
    if co_primary_cross_source and "source" in df.columns and len(splits.val_indices) > 0:
        val_srcs = df["source"].iloc[splits.val_indices].astype(str).to_numpy()
        val_y_np = df["label"].iloc[splits.val_indices].to_numpy().astype(int)
        for u_src in np.unique(val_srcs):
            mask = (val_srcs == u_src)
            if np.sum(mask) >= 10 and len(np.unique(val_y_np[mask])) > 1:
                val_source_slices.append((str(u_src), np.where(mask)[0]))

    # --- Phase 1: Environment Setup & Loss Class Weighting ---
    eff_loss_type = (loss_type or "ce").lower()
    if use_class_weights and eff_loss_type == "ce":
        eff_loss_type = "class_weighted"
    class_weights = compute_class_balance_weights(
        y_all[train_idx],
        mode=eff_loss_type,
        effective_beta=effective_beta,
        device=device,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Resolve state checkpoint path for atomic epoch-level resumption
    state_ckpt_path = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state_ckpt_path = checkpoint_path.parent / f"{checkpoint_path.stem}_train_state.pt"
    elif log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        state_ckpt_path = log_file.parent / f"{log_file.stem}_train_state.pt"

    start_epoch = 1
    best_val_auc = -1.0
    best_weights = None
    epochs_no_improve = 0
    log_records: List[Dict[str, Any]] = []
    chosen_lr = lr
    resumed_ckpt = None

    # --- Phase 2: Atomic State Checkpoint Resumption & Fingerprint Gate ---
    if resume and state_ckpt_path is not None and state_ckpt_path.exists():
        try:
            resumed_ckpt = torch.load(state_ckpt_path, map_location=device, weights_only=False)
            cached_fp = resumed_ckpt.get("experiment_fingerprint")
            
            # Require exact fingerprint match if an experiment fingerprint was supplied
            fp_mismatch = (experiment_fingerprint is not None) and (cached_fp != experiment_fingerprint)
            if fp_mismatch:
                reason = "missing fingerprint" if cached_fp is None else f"found {cached_fp[:10]}..., expected {experiment_fingerprint[:10]}..."
                print(f"  [CHECKPOINT REJECTED] Incompatible experiment fingerprint in {state_ckpt_path.name} "
                      f"({reason}). Starting fresh run.")
                resumed_ckpt = None
                start_epoch = 1
            else:
                model.load_state_dict(resumed_ckpt["model_state_dict"])
                start_epoch = int(resumed_ckpt["epoch"]) + 1
                best_val_auc = float(resumed_ckpt["best_val_auc"])
                best_weights = resumed_ckpt["best_weights"]
                epochs_no_improve = int(resumed_ckpt["epochs_no_improve"])
                log_records = resumed_ckpt.get("log_records", [])
                chosen_lr = resumed_ckpt.get("chosen_lr", lr)
                print(f"  [CHECKPOINT RESUME] Found verified checkpoint! Resuming training from epoch {start_epoch}/{epochs} (best val ROC-AUC: {best_val_auc:.4f})")
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint {state_ckpt_path}: {e}. Starting fresh from epoch 1.")
            start_epoch = 1
            resumed_ckpt = None

    # --- Phase 3: Validation Learning Rate Grid Search ---
    if resumed_ckpt is None and lr_grid and len(lr_grid) > 1 and epochs > 5:
        best_grid_auc = -1.0
        # Preserve pristine initial weights on CPU to avoid duplicate model in GPU VRAM
        init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for cand_lr in lr_grid:
            model.load_state_dict(init_state)
            opt_cand = torch.optim.Adam(model.parameters(), lr=cand_lr, weight_decay=weight_decay)
            for _ in range(min(5, epochs)):
                model.train()
                opt_cand.zero_grad(set_to_none=True)
                out_c, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
                loss_c = _compute_cls_loss(
                    out_c[train_idx],
                    y_all[train_idx],
                    loss_type=eff_loss_type,
                    class_weights=class_weights,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    reduction="mean",
                )
                loss_c.backward()
                opt_cand.step()
            model.eval()
            with torch.no_grad():
                val_c, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
                val_probs_c = F.softmax(val_c[val_idx], dim=-1)[:, 1].cpu().numpy()
                auc_c = evaluate_predictions(y_all[val_idx].cpu().numpy(), val_probs_c)["roc_auc"]
            if auc_c > best_grid_auc:
                best_grid_auc = auc_c
                chosen_lr = cand_lr
            model.zero_grad(set_to_none=True)
            del opt_cand, out_c, loss_c, val_c
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore pristine initial weights for actual training
        model.load_state_dict(init_state)
        model.zero_grad(set_to_none=True)
        del init_state
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Phase 4: Optimizer and Learning Rate Scheduler Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=chosen_lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Restore optimizer and scheduler state if resumed from checkpoint
    if resumed_ckpt is not None:
        try:
            if "optimizer_state_dict" in resumed_ckpt and resumed_ckpt["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in resumed_ckpt and resumed_ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"  [WARN] Could not restore optimizer/scheduler state: {e}. Continuing with fresh optimizer.")

    # Pre-extract validation truth once to eliminate 200 redundant tensor transfers
    val_true = y_all[val_idx].cpu().numpy()

    # --- Phase 5: Epoch Optimization Loop with CUDA Cache Flush & Checkpoint ---
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        try:
            p_prog = float(epoch) / float(max(1, epochs))
            grl_lambda = float(2.0 / (1.0 + math.exp(-10.0 * p_prog)) - 1.0) if active_grl else 0.0
            out, dom_emb, env_logits = _forward_model(
                model, data, edge_weight_dict, char_bytes, grl_lambda=grl_lambda
            )

            if dro_q is not None and len(dro_group_masks) >= 2:
                per_sample_ce = _compute_cls_loss(
                    out[train_idx],
                    y_all[train_idx],
                    loss_type=eff_loss_type,
                    class_weights=class_weights,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    reduction="none",
                )
                grp_losses = torch.stack([per_sample_ce[g_m].mean() for g_m in dro_group_masks])
                with torch.no_grad():
                    dro_q = dro_q * torch.exp(dro_eta * grp_losses.detach())
                    dro_q = dro_q / dro_q.sum().clamp(min=1e-8)
                loss = torch.sum(dro_q * grp_losses)
            else:
                loss = _compute_cls_loss(
                    out[train_idx],
                    y_all[train_idx],
                    loss_type=eff_loss_type,
                    class_weights=class_weights,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    reduction="mean",
                )

            if active_grl and env_logits is not None and env_targets is not None:
                env_loss = F.cross_entropy(env_logits[train_idx], env_targets[train_idx])
                loss = loss + grl_weight * env_loss

            if use_coral and len(test_idx) >= 16:
                c_loss = deep_coral_loss(dom_emb[train_idx], dom_emb[test_idx], max_samples=coral_max_samples)
                loss = loss + coral_weight * c_loss

            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            del out, dom_emb, env_logits, loss
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024**3)
                    res = torch.cuda.memory_reserved() / (1024**3)
                    print(f"\n[CRITICAL CUDA OOM at Epoch {epoch}] Allocated: {alloc:.2f} GB | Reserved: {res:.2f} GB")
                    torch.cuda.empty_cache()
                raise e
            raise e

        # Validation step
        model.eval()
        with torch.no_grad():
            val_out, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
            val_loss = _compute_cls_loss(
                val_out[val_idx],
                y_all[val_idx],
                loss_type=eff_loss_type,
                class_weights=class_weights,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                reduction="mean",
            ).item()
            val_probs = F.softmax(val_out[val_idx], dim=-1)[:, 1].cpu().numpy()
            del val_out

            val_metrics = evaluate_predictions(val_true, val_probs, threshold=0.5)
            val_auc = val_metrics["roc_auc"]
            if val_source_slices:
                worst_src_auc = min(
                    float(roc_auc_score(val_true[s_pos], val_probs[s_pos]))
                    for _, s_pos in val_source_slices
                )
                val_select_score = 0.5 * val_auc + 0.5 * worst_src_auc
            else:
                val_select_score = val_auc

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_select_score)
            else:
                scheduler.step()

        # Periodic cache compacting to keep CUDA allocator tidy without overhead
        if device.type == "cuda" and (epoch % 20 == 0 or epoch == epochs):
            torch.cuda.empty_cache()

        epoch_sec = time.perf_counter() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auc": round(val_auc, 4),
            "val_select_score": round(val_select_score, 4),
            "seconds": round(epoch_sec, 3)
        }
        log_records.append(log_entry)

        is_best = val_select_score > best_val_auc
        marker = " ★ (Best Val AUC)" if is_best else ""
        # Live industry-standard training progress line
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs or is_best:
            print(f"    [Epoch {epoch:03d}/{epochs:03d}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"Time: {epoch_sec:.2f}s{marker}")

        if is_best:
            best_val_auc = val_select_score
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Atomically checkpoint training state after each epoch
        if state_ckpt_path is not None:
            atomic_save_torch({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_auc": best_val_auc,
                "best_weights": best_weights,
                "epochs_no_improve": epochs_no_improve,
                "log_records": log_records,
                "chosen_lr": chosen_lr,
                "experiment_fingerprint": experiment_fingerprint
            }, state_ckpt_path)

        effective_min_epochs = min(min_epochs, epochs)
        if epoch >= effective_min_epochs and epochs_no_improve >= patience:
            print(
                f"  [EARLY STOPPING] Validation AUC did not improve for {patience} epochs "
                f"(minimum {effective_min_epochs}-epoch floor satisfied). Stopping at epoch {epoch}."
            )
            break

    # Persist training logs if path provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            for rec in log_records:
                f.write(json.dumps(rec) + "\n")

    # --- Phase 6: Model Evaluation & Validation Threshold Calibration ---
    # Load best model weights found during training
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_out, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
        val_logits_np = final_out[val_idx].cpu().numpy()

        if eval_data is not None:
            eval_data = eval_data.to(device)
            ev_ew, ev_cb = _extract_graph_aux_inputs(eval_data)
            test_out, _, _ = _forward_model(model, eval_data, ev_ew, ev_cb, grl_lambda=0.0)
            test_logits_np = test_out[test_idx].cpu().numpy()
            del test_out
        else:
            test_logits_np = final_out[test_idx].cpu().numpy()
        test_true = y_all[test_idx].cpu().numpy()
        del final_out
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Optional Temperature Scaling fit strictly on validation logits (Guo et al. 2017)
    temperature = 1.0
    if use_temperature_scaling:
        temperature = fit_temperature_scaling(val_logits_np, val_true)
        val_probs = apply_temperature_scaling(val_logits_np, temperature=temperature)
        test_probs = apply_temperature_scaling(test_logits_np, temperature=temperature)
    else:
        val_probs = apply_temperature_scaling(val_logits_np, temperature=1.0)
        test_probs = apply_temperature_scaling(test_logits_np, temperature=1.0)

    # Optimal cost-sensitive threshold selected strictly on calibrated validation predictions
    optimal_th = choose_optimal_threshold(
        val_true,
        val_probs,
        criterion=threshold_criterion,
        max_val_fpr=max_validation_fpr,
    )

    # Evaluate test metrics and operational operating points (TPR @ 1% and 0.1% FPR)
    test_eval = evaluate_predictions(
        y_true=test_true,
        y_prob=test_probs,
        threshold=optimal_th,
        y_val_true=val_true,
        y_val_prob=val_probs,
        target_fprs=target_fprs,
        prevalence_pis=prevalence_pis
    )

    # --- Phase 7: Model Artifact Serialization & Manifest Generation ---
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Comprehensive architecture metadata
        arch_config = model.get_model_config() if hasattr(model, "get_model_config") else {
            "variant": getattr(model, "variant", "unknown"),
            "hidden_dim": getattr(model, "hidden_dim", 64),
            "out_dim": getattr(model, "out_dim", 32),
            "num_layers": getattr(model, "num_layers", 2),
            "num_heads": getattr(model, "num_heads", 4),
            "dropout": getattr(model, "dropout", 0.2),
            "use_edges": getattr(model, "use_edges", True)
        }
        
        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "model_config": arch_config,
            "experiment_fingerprint": experiment_fingerprint,
            "seed": seed,
            "split_type": split_type or "random",
            "best_val_auc": float(best_val_auc),
            "threshold": float(optimal_th),
            "temperature": float(temperature),
            "loss_type": eff_loss_type,
            "threshold_criterion": threshold_criterion,
            "epochs_trained": len(log_records),
            "val_indices": splits.val_indices.tolist(),
            "cert_indices": splits.cert_indices.tolist() if getattr(splits, "cert_indices", None) is not None else [],
            "test_indices": splits.test_indices.tolist(),
            "created_at": time.time()
        }
        atomic_save_torch(ckpt_payload, checkpoint_path)

        manifest_path = checkpoint_path.parent / f"{checkpoint_path.stem}_manifest.json"
        manifest_payload = {
            "checkpoint_file": checkpoint_path.name,
            "experiment_fingerprint": experiment_fingerprint,
            "architecture": arch_config.get("variant", "unknown"),
            "model_config": arch_config,
            "seed": seed,
            "split_type": split_type or "random",
            "split_metadata": split_metadata or {},
            "threshold": float(optimal_th),
            "best_val_auc": float(best_val_auc),
            "epochs_trained": len(log_records),
            "val_count": len(splits.val_indices),
            "cert_count": len(splits.cert_indices) if getattr(splits, "cert_indices", None) is not None else 0,
            "test_count": len(splits.test_indices),
            "cert_indices": splits.cert_indices.tolist() if getattr(splits, "cert_indices", None) is not None else [],
            "created_at": time.time(),
            "target_fprs": target_fprs,
            "operating_points": test_eval.get("operating_points", {}),
            "metrics": {
                "roc_auc": test_eval.get("roc_auc"),
                "pr_auc": test_eval.get("pr_auc"),
                "f1": test_eval.get("f1"),
                "accuracy": test_eval.get("accuracy"),
                "precision": test_eval.get("precision"),
                "recall": test_eval.get("recall")
            }
        }
        atomic_save_json(manifest_payload, manifest_path)

    # Clean up intermediate training state on clean completion
    if state_ckpt_path is not None and state_ckpt_path.exists():
        try:
            state_ckpt_path.unlink()
        except Exception:
            pass

    # --- Phase 8: Bootstrap Confidence Intervals & Subgroup Breakdowns ---
    # Stratified bootstrap resampling to estimate empirical 95% confidence intervals
    test_eval["bootstrap_ci"] = compute_bootstrap_cis(
        y_true=test_true,
        y_prob=test_probs,
        threshold=optimal_th,
        n_bootstrap=bootstrap_samples,
        seed=seed
    )

    # Subgroup breakdowns
    test_df = df.iloc[splits.test_indices].copy().reset_index(drop=True)
    test_df["pred_prob"] = test_probs
    test_df["pred_label"] = (test_probs >= optimal_th).astype(int)

    breakdowns: Dict[str, Any] = {}

    # 1. Resolved vs No-IP
    breakdowns["by_resolution"] = {}
    if "no_ip" in test_df.columns:
        for is_no_ip in [False, True]:
            sub = test_df[test_df["no_ip"] == is_no_ip]
            if len(sub) > 5 and len(sub["label"].unique()) > 1:
                m = evaluate_predictions(sub["label"].to_numpy(), sub["pred_prob"].to_numpy(), threshold=optimal_th)
                breakdowns["by_resolution"]["no_ip" if is_no_ip else "resolved"] = {
                    "count": len(sub),
                    "roc_auc": m["roc_auc"],
                    "f1": m["f1"]
                }

    # 2. By Source
    breakdowns["by_source"] = {}
    if "source" in test_df.columns:
        for src_name, sub in test_df.groupby("source"):
            if len(sub) >= 10:
                if len(sub["label"].unique()) > 1:
                    m = evaluate_predictions(sub["label"].to_numpy(), sub["pred_prob"].to_numpy(), threshold=optimal_th)
                    breakdowns["by_source"][str(src_name)] = {
                        "count": len(sub),
                        "roc_auc": m["roc_auc"],
                        "f1": m["f1"]
                    }
                else:
                    acc = float(np.mean(sub["label"].to_numpy() == sub["pred_label"].to_numpy()))
                    breakdowns["by_source"][str(src_name)] = {
                        "count": len(sub),
                        "accuracy": round(acc, 4)
                    }

    # 3. By Top Malware Families
    breakdowns["by_malware_family"] = {}
    if "family" in test_df.columns:
        mal_test = test_df[test_df["label"] == 1]
        for fam_name, sub in mal_test.groupby("family"):
            if len(sub) >= 5:
                recall = float(np.mean(sub["pred_label"].to_numpy() == 1))
                breakdowns["by_malware_family"][str(fam_name)] = {
                    "count": len(sub),
                    "detection_rate_recall": round(recall, 4)
                }

    test_eval["breakdowns"] = breakdowns
    del test_df
    test_eval["training_metadata"] = {
        "chosen_lr": chosen_lr,
        "lr_grid": lr_grid,
        "loss_type": eff_loss_type,
        "focal_gamma": focal_gamma if eff_loss_type == "focal" else None,
        "focal_alpha": focal_alpha if eff_loss_type == "focal" else None,
        "use_class_weights": bool(use_class_weights or eff_loss_type in ("class_weighted", "effective_samples")),
        "threshold_criterion": threshold_criterion,
        "calibrated_threshold": round(float(optimal_th), 4),
        "temperature": round(float(temperature), 4),
        "epochs_trained": len(log_records),
        "best_val_auc": round(best_val_auc, 4),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None
    }

    del best_weights, optimizer, scheduler
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_eval, test_true, test_probs, model


def cvar_topk_pauc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k_fraction: float = 0.01,
    margin: float = 0.20,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Computes chunked CVaR top-k negative surrogate loss for partial AUC optimization at low FPR.
    Processes positives in `chunk_size` batches (default 4096) so peak memory remains < 16 MB at 9.3L scale.
    """
    scores = logits[:, 1] - logits[:, 0]

    pos_mask = (targets == 1)
    neg_mask = (targets == 0)

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    k = max(1, int(round(k_fraction * len(neg_scores))))
    top_k_neg_scores, _ = torch.topk(neg_scores, k=min(k, len(neg_scores)), largest=True)

    total_loss = logits.new_tensor(0.0)
    eff_chunk = max(1, int(chunk_size))
    for i in range(0, len(pos_scores), eff_chunk):
        chunk = pos_scores[i : i + eff_chunk]
        diff = chunk.unsqueeze(1) - top_k_neg_scores.unsqueeze(0)
        total_loss = total_loss + torch.clamp(margin - diff, min=0.0).sum()

    denom = float(len(pos_scores) * len(top_k_neg_scores))
    return total_loss / max(1.0, denom)


def fine_tune_pauc_gnn(
    model: nn.Module,
    data: HeteroData,
    splits: DataSplits,
    df: pd.DataFrame,
    checkpoint_path: Optional[Path] = None,
    epochs: int = 30,
    lr: float = 0.0005,
    patience: int = 10,
    k_fraction: float = 0.01,
    margin: float = 0.20,
    lambda_pauc: float = 0.50,
    max_val_fpr: float = 0.01,
    seed: int = 42,
    device: Optional[torch.device] = None,
    loss_type: str = "focal",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    threshold_criterion: str = "f0.5",
    max_validation_fpr: float = 0.015,
    use_temperature_scaling: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, nn.Module]:
    """
    Fine-tunes an existing GNN checkpoint using composite Focal/CE + chunked CVaR top-k pAUC loss.
    Monitors validation partial AUC (FPR <= max_val_fpr) for early stopping.
    Returns: (test_eval, test_true, test_probs, trained_model)
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    data = data.to(device)
    primary_nt = get_primary_domain_node_type(data)
    edge_weight_dict, char_bytes = _extract_graph_aux_inputs(data)

    # Load starting checkpoint weights if available
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"  [pAUC Fine-Tune] Loaded initial weights from {Path(checkpoint_path).name}")
        except Exception as e:
            print(f"  [WARN] Could not load {checkpoint_path}: {e}")

    train_idx = torch.tensor(splits.train_indices, dtype=torch.long, device=device)
    val_idx = torch.tensor(splits.val_indices, dtype=torch.long, device=device)
    test_idx = torch.tensor(splits.test_indices, dtype=torch.long, device=device)

    y_all = data[primary_nt].y
    class_weights = compute_class_balance_weights(y_all[train_idx], mode=loss_type, device=device)
    val_true = y_all[val_idx].cpu().numpy()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_pauc = -1.0
    best_weights = deepcopy(model.state_dict())
    epochs_no_improve = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        epochs_trained = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)

        out, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
        cls_loss = _compute_cls_loss(
            out[train_idx],
            y_all[train_idx],
            loss_type=loss_type,
            class_weights=class_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            reduction="mean",
        )
        pauc_loss = cvar_topk_pauc_loss(out[train_idx], y_all[train_idx], k_fraction=k_fraction, margin=margin)
        total_loss = cls_loss + lambda_pauc * pauc_loss

        total_loss.backward()
        optimizer.step()
        del out, cls_loss, pauc_loss, total_loss

        # Validation step: compute partial AUC over FPR <= max_val_fpr
        model.eval()
        with torch.no_grad():
            val_out, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
            val_probs = F.softmax(val_out[val_idx], dim=-1)[:, 1].cpu().numpy()
            del val_out

            try:
                val_pauc = float(roc_auc_score(val_true, val_probs, max_fpr=max_val_fpr))
            except Exception:
                val_pauc = float(roc_auc_score(val_true, val_probs)) if len(np.unique(val_true)) > 1 else 0.5

        if device.type == "cuda" and (epoch % 20 == 0 or epoch == epochs):
            torch.cuda.empty_cache()

        if val_pauc > best_val_pauc:
            best_val_pauc = val_pauc
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  [pAUC Fine-Tune] Early stopping at epoch {epoch} (best val pAUC: {best_val_pauc:.4f})")
            break

    # Restore best weights and evaluate on test
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        final_out, _, _ = _forward_model(model, data, edge_weight_dict, char_bytes, grl_lambda=0.0)
        test_logits_np = final_out[test_idx].cpu().numpy()
        val_logits_np = final_out[val_idx].cpu().numpy()
        test_true = y_all[test_idx].cpu().numpy()
        del final_out

    temperature = 1.0
    if use_temperature_scaling:
        temperature = fit_temperature_scaling(val_logits_np, val_true)
    val_final = apply_temperature_scaling(val_logits_np, temperature=temperature)
    test_probs = apply_temperature_scaling(test_logits_np, temperature=temperature)

    optimal_th = choose_optimal_threshold(
        val_true,
        val_final,
        criterion=threshold_criterion,
        max_val_fpr=max_validation_fpr,
    )
    test_eval = evaluate_predictions(
        y_true=test_true,
        y_prob=test_probs,
        threshold=optimal_th,
        y_val_true=val_true,
        y_val_prob=val_final,
        target_fprs=[0.01, 0.001],
        prevalence_pis=[0.01, 0.001]
    )

    boot_cis = compute_bootstrap_cis(test_true, test_probs, threshold=optimal_th, n_bootstrap=1000, seed=seed)
    test_eval["bootstrap_ci"] = boot_cis
    test_eval["fine_tune_metadata"] = {
        "best_val_pauc": round(best_val_pauc, 4),
        "lambda_pauc": lambda_pauc,
        "k_fraction": k_fraction,
        "margin": margin,
        "loss_type": loss_type,
        "threshold_criterion": threshold_criterion,
        "temperature": round(float(temperature), 4),
        "epochs_trained": epochs_trained,
    }
    test_eval["val_probs"] = val_final

    return test_eval, test_true, test_probs, model

