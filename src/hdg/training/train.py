"""
Unified, leak-free training and evaluation loop for Malicious-HDG GNN models.
Replaces legacy copy-pasted loops from scripts 07, 10, 12, 13, 14, 17, 18.
- Validation-driven learning rate search (lr_grid)
- Early stopping on validation ROC-AUC
- Optional class weights and learning rate scheduler
- Per-epoch training logging to results/logs/
- Model checkpointing to models/checkpoints/
- Threshold optimization strictly on validation set
- Operating points (TPR at 1% and 0.1% FPR) with validation negative thresholds
- Realised test FPR and prevalence-adjusted precisions
- Bootstrap 95% confidence intervals
- Subgroup breakdowns (resolved vs no_ip, source, malware family)
"""

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.metrics import (
    compute_bootstrap_cis,
    choose_optimal_threshold,
    evaluate_predictions,
)
from src.hdg.data.splits import DataSplits


def train_eval_gnn(
    model: nn.Module,
    data: HeteroData,
    splits: DataSplits,
    df: pd.DataFrame,
    epochs: int = 200,
    lr: float = 0.005,
    lr_grid: Optional[List[float]] = None,
    weight_decay: float = 0.0001,
    patience: int = 20,
    target_fprs: List[float] = [0.01, 0.001],
    prevalence_pis: List[float] = [0.01, 0.001],
    bootstrap_samples: int = 1000,
    use_class_weights: bool = False,
    scheduler_type: Optional[str] = "plateau",
    checkpoint_path: Optional[Path] = None,
    log_file: Optional[Path] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
    resume: bool = True
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, nn.Module]:
    """
    Trains HeteroGNN with early stopping on validation ROC-AUC and validation-tuned threshold.
    Supports atomic epoch-level checkpointing and resumption so long-running training
    (e.g. multi-day CPU runs) can resume immediately after interrupts without starting from scratch.
    Returns: (eval_results_dict, y_test_true, y_test_prob, trained_model)
    """
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    data = data.to(device)

    train_idx = torch.tensor(splits.train_indices, dtype=torch.long, device=device)
    val_idx = torch.tensor(splits.val_indices, dtype=torch.long, device=device)
    test_idx = torch.tensor(splits.test_indices, dtype=torch.long, device=device)

    y_all = data["domain"].y

    # Calculate class weights if requested
    class_weights = None
    if use_class_weights:
        y_train = y_all[train_idx]
        pos_cnt = int((y_train == 1).sum().item())
        neg_cnt = int((y_train == 0).sum().item())
        if pos_cnt > 0 and neg_cnt > 0:
            weight_1 = float(neg_cnt / pos_cnt)
            class_weights = torch.tensor([1.0, weight_1], dtype=torch.float, device=device)

    # Resolve state checkpoint path for safe resumption
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

    # Attempt checkpoint resumption
    if resume and state_ckpt_path is not None and state_ckpt_path.exists():
        try:
            resumed_ckpt = torch.load(state_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(resumed_ckpt["model_state_dict"])
            start_epoch = int(resumed_ckpt["epoch"]) + 1
            best_val_auc = float(resumed_ckpt["best_val_auc"])
            best_weights = resumed_ckpt["best_weights"]
            epochs_no_improve = int(resumed_ckpt["epochs_no_improve"])
            log_records = resumed_ckpt.get("log_records", [])
            chosen_lr = resumed_ckpt.get("chosen_lr", lr)
            print(f"  [CHECKPOINT RESUME] Found existing checkpoint! Resuming training from epoch {start_epoch}/{epochs} (best val ROC-AUC: {best_val_auc:.4f})")
        except Exception as e:
            print(f"  [WARN] Failed to load checkpoint {state_ckpt_path}: {e}. Starting fresh from epoch 1.")
            start_epoch = 1
            resumed_ckpt = None

    # Validation learning rate search if grid provided (only if starting fresh)
    if resumed_ckpt is None and lr_grid and len(lr_grid) > 1 and epochs > 5:
        best_grid_auc = -1.0
        for cand_lr in lr_grid:
            m_copy = deepcopy(model)
            opt_cand = torch.optim.Adam(m_copy.parameters(), lr=cand_lr, weight_decay=weight_decay)
            for _ in range(min(5, epochs)):
                m_copy.train()
                opt_cand.zero_grad()
                out_c = m_copy(data.x_dict, data.edge_index_dict)
                loss_c = F.cross_entropy(out_c[train_idx], y_all[train_idx], weight=class_weights)
                loss_c.backward()
                opt_cand.step()
            m_copy.eval()
            with torch.no_grad():
                val_c = m_copy(data.x_dict, data.edge_index_dict)
                val_probs_c = F.softmax(val_c[val_idx], dim=-1)[:, 1].cpu().numpy()
                auc_c = evaluate_predictions(y_all[val_idx].cpu().numpy(), val_probs_c)["roc_auc"]
            if auc_c > best_grid_auc:
                best_grid_auc = auc_c
                chosen_lr = cand_lr
            del m_copy, opt_cand
            import gc
            gc.collect()

    optimizer = torch.optim.Adam(model.parameters(), lr=chosen_lr, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Restore optimizer and scheduler state if resumed
    if resumed_ckpt is not None:
        try:
            if "optimizer_state_dict" in resumed_ckpt and resumed_ckpt["optimizer_state_dict"] is not None:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in resumed_ckpt and resumed_ckpt["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"  [WARN] Could not restore optimizer/scheduler state: {e}. Continuing with fresh optimizer.")

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            val_loss = F.cross_entropy(val_out[val_idx], y_all[val_idx], weight=class_weights).item()
            val_probs = F.softmax(val_out[val_idx], dim=-1)[:, 1].cpu().numpy()
            val_true = y_all[val_idx].cpu().numpy()

            val_metrics = evaluate_predictions(val_true, val_probs, threshold=0.5)
            val_auc = val_metrics["roc_auc"]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        epoch_sec = time.perf_counter() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auc": round(val_auc, 4),
            "seconds": round(epoch_sec, 3)
        }
        log_records.append(log_entry)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Atomically checkpoint training state after each epoch
        if state_ckpt_path is not None:
            tmp_path = state_ckpt_path.with_suffix(".tmp")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "best_val_auc": best_val_auc,
                "best_weights": best_weights,
                "epochs_no_improve": epochs_no_improve,
                "log_records": log_records,
                "chosen_lr": chosen_lr,
            }, tmp_path)
            tmp_path.replace(state_ckpt_path)

        if epochs_no_improve >= patience:
            print(f"  [EARLY STOPPING] Validation AUC did not improve for {patience} epochs. Stopping at epoch {epoch}.")
            break

    # Persist training logs if path provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            for rec in log_records:
                f.write(json.dumps(rec) + "\n")

    # Load best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save finalized model checkpoint
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    # Clean up intermediate training state on clean completion
    if state_ckpt_path is not None and state_ckpt_path.exists():
        try:
            state_ckpt_path.unlink()
        except Exception:
            pass

    model.eval()
    with torch.no_grad():
        final_out = model(data.x_dict, data.edge_index_dict)
        val_probs = F.softmax(final_out[val_idx], dim=-1)[:, 1].cpu().numpy()
        val_true = y_all[val_idx].cpu().numpy()

        test_probs = F.softmax(final_out[test_idx], dim=-1)[:, 1].cpu().numpy()
        test_true = y_all[test_idx].cpu().numpy()

    # Select threshold on validation set
    optimal_th = choose_optimal_threshold(val_true, val_probs)

    # Evaluate on test set
    test_eval = evaluate_predictions(
        y_true=test_true,
        y_prob=test_probs,
        threshold=optimal_th,
        y_val_true=val_true,
        y_val_prob=val_probs,
        target_fprs=target_fprs,
        prevalence_pis=prevalence_pis
    )

    # Bootstrap confidence intervals on test set
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
    mal_test = test_df[test_df["label"] == 1]
    for fam_name, sub in mal_test.groupby("family"):
        if len(sub) >= 5:
            recall = float(np.mean(sub["pred_label"].to_numpy() == 1))
            breakdowns["by_malware_family"][str(fam_name)] = {
                "count": len(sub),
                "detection_rate_recall": round(recall, 4)
            }

    test_eval["breakdowns"] = breakdowns
    test_eval["training_metadata"] = {
        "chosen_lr": chosen_lr,
        "lr_grid": lr_grid,
        "epochs_trained": len(log_records),
        "best_val_auc": round(best_val_auc, 4),
        "use_class_weights": use_class_weights,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None
    }

    del best_weights, optimizer, scheduler
    import gc
    gc.collect()

    return test_eval, test_true, test_probs, model
