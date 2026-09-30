"""
Unified, leak-free training and evaluation loop for Malicious-HDG GNN models.
Replaces legacy copy-pasted loops from scripts 07, 10, 12, 13, 14, 17, 18.
- Early stopping on validation ROC-AUC
- Threshold optimization on validation set only
- Operating points (TPR at 1% and 0.1% FPR) with validation negative thresholds
- Bootstrap 95% confidence intervals
- Subgroup breakdowns (resolved vs no_ip, source, malware family)
"""

from copy import deepcopy
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
from src.hdg.splits import DataSplits


def train_eval_gnn(
    model: nn.Module,
    data: HeteroData,
    splits: DataSplits,
    df: pd.DataFrame,
    epochs: int = 30,
    lr: float = 0.005,
    weight_decay: float = 0.0001,
    patience: int = 5,
    target_fprs: List[float] = [0.01, 0.001],
    bootstrap_samples: int = 1000,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, nn.Module]:
    """
    Trains HeteroGNN with early stopping on validation ROC-AUC.
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_weights = None
    epochs_no_improve = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_out = model(data.x_dict, data.edge_index_dict)
            val_loss = F.cross_entropy(val_out[val_idx], y_all[val_idx]).item()
            val_probs = F.softmax(val_out[val_idx], dim=-1)[:, 1].cpu().numpy()
            val_true = y_all[val_idx].cpu().numpy()

            val_metrics = evaluate_predictions(val_true, val_probs, threshold=0.5)
            val_auc = val_metrics["roc_auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    # Load best model weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

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
        target_fprs=target_fprs
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
                # Single class slice (e.g. benign source or pure malware source)
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

    return test_eval, test_true, test_probs, model
