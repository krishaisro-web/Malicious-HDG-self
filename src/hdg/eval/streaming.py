"""
Streaming domain evaluation for online updates in Malicious-HDG.
Implements:
1. Reservoir sampling buffer for domain replay.
2. Continual learning / streaming evaluation comparing:
   - Retrain from scratch (cumulative historical data, upper bound)
   - Naive fine-tuning (online training on new month; vulnerable to catastrophic forgetting)
   - Replay buffer (reservoir sample of size 1000 mixed with incoming month)
3. Evaluates forward performance (F1, TPR@0.1% FPR, ROC-AUC) on new month T+1.
4. Evaluates backward transfer / forgetting on month 1 at each incremental step.
"""

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.metrics import choose_optimal_threshold, evaluate_predictions
from src.hdg.data.splits import DataSplits

logger = logging.getLogger(__name__)


class ReservoirBuffer:
    """Fixed-capacity reservoir sampling buffer for historical training domains."""

    def __init__(self, capacity: int = 1000, seed: int = 42) -> None:
        self.capacity = capacity
        self.buffer: List[int] = []
        self.total_seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, indices: np.ndarray) -> None:
        for idx in indices:
            self.total_seen += 1
            if len(self.buffer) < self.capacity:
                self.buffer.append(int(idx))
            else:
                j = self.rng.integers(0, self.total_seen)
                if j < self.capacity:
                    self.buffer[j] = int(idx)

    def get_sample(self, n: Optional[int] = None) -> np.ndarray:
        if not self.buffer:
            return np.empty(0, dtype=int)
        if n is None or n >= len(self.buffer):
            return np.array(self.buffer, dtype=int)
        sampled = self.rng.choice(self.buffer, size=n, replace=False)
        return np.array(sampled, dtype=int)

    def __len__(self) -> int:
        return len(self.buffer)


def evaluate_on_indices(
    model: nn.Module,
    data: HeteroData,
    eval_indices: np.ndarray,
    y_true_all: np.ndarray,
    threshold: float = 0.5,
    val_indices: Optional[np.ndarray] = None,
    calibrated_threshold_01: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluates model on specified domain indices.
    Operating point TPR @ 0.1% FPR is evaluated using a frozen validation-calibrated threshold.
    Test labels are never used to choose the operating point.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out[eval_indices], dim=-1)[:, 1].cpu().numpy()

    y_eval = y_true_all[eval_indices]
    if len(np.unique(y_eval)) < 2:
        auc = 0.5
        tpr_at_01 = 0.0
        rfpr_at_01 = 0.0
        chosen_th_01 = calibrated_threshold_01 if calibrated_threshold_01 is not None else 0.5
    else:
        auc = float(roc_auc_score(y_eval, probs))
        
        # Strictly determine operating threshold using validation negatives, NEVER test labels
        if calibrated_threshold_01 is not None:
            chosen_th_01 = float(calibrated_threshold_01)
        elif val_indices is not None and len(val_indices) > 0:
            with torch.no_grad():
                v_probs = F.softmax(out[val_indices], dim=-1)[:, 1].cpu().numpy()
            v_negs = v_probs[y_true_all[val_indices] == 0]
            if len(v_negs) > 0:
                sorted_v_negs = np.sort(v_negs)
                cutoff_idx = int(np.ceil((1.0 - 0.001) * (len(sorted_v_negs) - 1)))
                chosen_th_01 = float(sorted_v_negs[cutoff_idx])
            else:
                chosen_th_01 = 0.5
        else:
            chosen_th_01 = 0.5

        # Evaluate realized metrics on the test cohort using the frozen calibrated threshold
        test_pos = probs[y_eval == 1]
        test_negs = probs[y_eval == 0]
        tpr_at_01 = float(np.mean(test_pos >= chosen_th_01)) if len(test_pos) > 0 else 0.0
        rfpr_at_01 = float(np.mean(test_negs >= chosen_th_01)) if len(test_negs) > 0 else 0.0

    y_pred = (probs >= threshold).astype(int)
    f1 = float(f1_score(y_eval, y_pred, zero_division=0))
    prec = float(precision_score(y_eval, y_pred, zero_division=0))
    rec = float(recall_score(y_eval, y_pred, zero_division=0))

    return {
        "roc_auc": round(auc, 4),
        "f1": round(f1, 4),
        "tpr_at_01_fpr": round(tpr_at_01, 4),
        "realized_fpr_at_01": round(rfpr_at_01, 6),
        "calibrated_threshold_01": round(float(chosen_th_01), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "threshold": round(float(threshold), 4),
        "num_samples": len(eval_indices)
    }


def train_step(
    model: nn.Module,
    data: HeteroData,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    y_tensor: torch.Tensor,
    y_true_all: np.ndarray,
    epochs: int = 20,
    lr: float = 0.005,
    weight_decay: float = 0.0001
) -> Tuple[nn.Module, float]:
    """Runs a training loop and returns tuned threshold based on validation set."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_weights = deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(out[train_indices], y_tensor[train_indices])
        loss.backward()
        optimizer.step()

        # Simple validation loss tracking
        model.eval()
        with torch.no_grad():
            v_out = model(data.x_dict, data.edge_index_dict)
            v_loss = F.cross_entropy(v_out[val_indices], y_tensor[val_indices]).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        v_out = model(data.x_dict, data.edge_index_dict)
        v_probs = F.softmax(v_out[val_indices], dim=-1)[:, 1].cpu().numpy()
        v_true = y_true_all[val_indices]

    if len(np.unique(v_true)) >= 2:
        val_thresh = choose_optimal_threshold(v_true, v_probs)
    else:
        val_thresh = 0.5

    return model, val_thresh


from src.hdg.data.graph import apply_split_scaling


def calibrate_operating_threshold(
    model: nn.Module,
    data: HeteroData,
    val_indices: np.ndarray,
    y_true_all: np.ndarray,
    target_fpr: float = 0.001
) -> float:
    """Calibrates low-FPR operating threshold strictly on validation negatives."""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out[val_indices], dim=-1)[:, 1].cpu().numpy()
    val_negs = probs[y_true_all[val_indices] == 0]
    if len(val_negs) == 0:
        return 0.5
    sorted_v_negs = np.sort(val_negs)
    cutoff_idx = int(np.ceil((1.0 - target_fpr) * (len(sorted_v_negs) - 1)))
    return float(sorted_v_negs[cutoff_idx])


def run_streaming_evaluation(
    data: HeteroData,
    df: pd.DataFrame,
    temporal_col: str = "t_month",
    replay_reservoir_size: int = 1000,
    epochs_scratch: int = 40,
    epochs_finetune: int = 15,
    max_months: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes incremental streaming evaluation over consecutive chronological months.
    Applies split scaling and cutoff-appropriate topology masking (mask_future_domains=True,
    degree_mode='train_visible') for each step.
    Operating point at 0.1% FPR is strictly calibrated on historical validation negatives
    and frozen prior to evaluation on test cohorts.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Identify valid months
    unique_months = sorted(df[temporal_col].dropna().unique().tolist())
    valid_months = []
    month_data = {}

    for m in unique_months:
        m_idx = df.index[df[temporal_col] == m].to_numpy()
        m_labels = df.loc[m_idx, "label"].to_numpy()
        # Require both classes
        if len(np.unique(m_labels)) >= 2 and len(m_idx) >= 10:
            valid_months.append(m)
            # Stratified 70/15/15 split within month
            pos_idx = m_idx[m_labels == 1]
            neg_idx = m_idx[m_labels == 0]
            rng.shuffle(pos_idx)
            rng.shuffle(neg_idx)

            n_pos_train = max(1, int(len(pos_idx) * 0.70))
            n_pos_val = max(1, int(len(pos_idx) * 0.15))
            n_neg_train = max(1, int(len(neg_idx) * 0.70))
            n_neg_val = max(1, int(len(neg_idx) * 0.15))

            tr = np.concatenate([pos_idx[:n_pos_train], neg_idx[:n_neg_train]])
            va = np.concatenate([pos_idx[n_pos_train:n_pos_train+n_pos_val], neg_idx[n_neg_train:n_neg_train+n_neg_val]])
            te = np.concatenate([pos_idx[n_pos_train+n_pos_val:], neg_idx[n_neg_train+n_neg_val:]])

            rng.shuffle(tr)
            rng.shuffle(va)
            rng.shuffle(te)

            month_data[m] = {
                "train": tr,
                "val": va,
                "test": te
            }

    if max_months is not None and len(valid_months) > max_months:
        valid_months = valid_months[:max_months]

    if len(valid_months) < 2:
        month_class_counts = {}
        for m in unique_months:
            m_labels = df.loc[df[temporal_col] == m, "label"].to_numpy()
            month_class_counts[str(m)] = {
                "total": int(len(m_labels)),
                "malware": int((m_labels == 1).sum()),
                "benign": int((m_labels == 0).sum())
            }
        logger.warning(
            f"Streaming evaluation cannot proceed: need >= 2 valid months with both classes, found {len(valid_months)}."
        )
        return {
            "status": "BLOCKED_INSUFFICIENT_DATA",
            "reason": f"Need at least 2 valid months with both classes for continual learning, but found {len(valid_months)}.",
            "valid_months": valid_months,
            "all_months": unique_months,
            "month_class_counts": month_class_counts,
            "temporal_column": temporal_col,
            "steps": [],
            "message": "Streaming evaluation blocked due to lack of multiple consecutive multi-class months."
        }

    y_tensor = data["domain"].y
    y_true_all = y_tensor.cpu().numpy()

    # Month 0 initialization
    m0 = valid_months[0]
    m0_splits = month_data[m0]

    # Build cutoff-appropriate scaled graph for Month 0
    # Future months' domains are masked from message passing
    all_future_domains = np.concatenate([month_data[m]["train"] for m in valid_months[1:]] +
                                        [month_data[m]["val"] for m in valid_months[1:]] +
                                        [month_data[m]["test"] for m in valid_months[1:]] +
                                        [m0_splits["test"]])
    m0_scaled_data, _ = apply_split_scaling(
        base=data,
        train_indices=m0_splits["train"],
        val_indices=m0_splits["val"],
        test_indices=all_future_domains,
        degree_mode="train_visible",
        mask_future_domains=True
    )

    in_channels = {nt: m0_scaled_data[nt].x.shape[1] for nt in m0_scaled_data.node_types}

    def build_fresh_model() -> HeteroGNN:
        return HeteroGNN(
            metadata=m0_scaled_data.metadata(),
            in_channels_dict=in_channels,
            variant="sage",
            hidden_dim=64,
            out_dim=32,
            num_layers=2
        )

    # Train base model on Month 0 using scaled, cutoff graph
    base_model = build_fresh_model()
    base_model, m0_thresh = train_step(
        model=base_model,
        data=m0_scaled_data,
        train_indices=m0_splits["train"],
        val_indices=m0_splits["val"],
        y_tensor=y_tensor,
        y_true_all=y_true_all,
        epochs=epochs_scratch
    )

    # Calibrate operating threshold at 0.1% FPR strictly on Month 0 validation negatives
    m0_calibrated_th_01 = calibrate_operating_threshold(
        base_model, m0_scaled_data, m0_splits["val"], y_true_all, target_fpr=0.001
    )

    # Evaluate Month 0 baseline using frozen calibrated threshold
    m0_baseline_perf = evaluate_on_indices(
        base_model, m0_scaled_data, m0_splits["test"], y_true_all,
        threshold=m0_thresh,
        val_indices=m0_splits["val"],
        calibrated_threshold_01=m0_calibrated_th_01
    )

    # Models for the 3 strategies
    model_finetune = deepcopy(base_model)
    model_replay = deepcopy(base_model)

    replay_buffer = ReservoirBuffer(capacity=replay_reservoir_size, seed=seed)
    replay_buffer.add(m0_splits["train"])

    cumulative_train = [m0_splits["train"]]
    cumulative_val = [m0_splits["val"]]

    # Historical validation negatives buffer for freezing operating threshold
    last_val_indices = m0_splits["val"]
    last_calibrated_th_01 = m0_calibrated_th_01

    step_results: List[Dict[str, Any]] = []

    # Record initial step 0
    step_results.append({
        "step": 0,
        "month": m0,
        "strategy": "initial_base",
        "current_month_perf": m0_baseline_perf,
        "month_1_perf": m0_baseline_perf,
        "backward_transfer_f1": 0.0
    })

    # Incremental steps
    for step_idx, m_cur in enumerate(valid_months[1:], start=1):
        cur_splits = month_data[m_cur]
        cumulative_train.append(cur_splits["train"])
        cumulative_val.append(cur_splits["val"])

        # Determine cutoff test set for this step: test cohort + any subsequent future months
        future_months = valid_months[step_idx+1:]
        step_future_domains = [cur_splits["test"]]
        for fm in future_months:
            step_future_domains.extend([month_data[fm]["train"], month_data[fm]["val"], month_data[fm]["test"]])
        step_test_set = np.concatenate(step_future_domains)

        all_tr = np.concatenate(cumulative_train)
        all_va = np.concatenate(cumulative_val)

        # Build cutoff-appropriate scaled graph for current step
        step_scaled_data, _ = apply_split_scaling(
            base=data,
            train_indices=all_tr,
            val_indices=all_va,
            test_indices=step_test_set,
            degree_mode="train_visible",
            mask_future_domains=True
        )

        # Operating point threshold for month m_cur MUST be calibrated strictly on historical validation negatives
        # available PRIOR to testing month m_cur (freeze threshold before evaluation)
        frozen_th_01 = last_calibrated_th_01

        # 1. Strategy: retrain_from_scratch
        model_scratch = build_fresh_model()
        model_scratch, scratch_thresh = train_step(
            model=model_scratch,
            data=step_scaled_data,
            train_indices=all_tr,
            val_indices=all_va,
            y_tensor=y_tensor,
            y_true_all=y_true_all,
            epochs=epochs_scratch
        )
        scratch_cur = evaluate_on_indices(
            model_scratch, step_scaled_data, cur_splits["test"], y_true_all,
            threshold=scratch_thresh,
            calibrated_threshold_01=frozen_th_01
        )
        scratch_m0 = evaluate_on_indices(
            model_scratch, step_scaled_data, m0_splits["test"], y_true_all,
            threshold=scratch_thresh,
            calibrated_threshold_01=m0_calibrated_th_01
        )
        bwt_scratch = round(scratch_m0["f1"] - m0_baseline_perf["f1"], 4)

        step_results.append({
            "step": step_idx,
            "month": m_cur,
            "strategy": "retrain_from_scratch",
            "current_month_perf": scratch_cur,
            "month_1_perf": scratch_m0,
            "backward_transfer_f1": bwt_scratch
        })

        # 2. Strategy: naive_finetune
        model_finetune, ft_thresh = train_step(
            model=model_finetune,
            data=step_scaled_data,
            train_indices=cur_splits["train"],
            val_indices=cur_splits["val"],
            y_tensor=y_tensor,
            y_true_all=y_true_all,
            epochs=epochs_finetune
        )
        ft_cur = evaluate_on_indices(
            model_finetune, step_scaled_data, cur_splits["test"], y_true_all,
            threshold=ft_thresh,
            calibrated_threshold_01=frozen_th_01
        )
        ft_m0 = evaluate_on_indices(
            model_finetune, step_scaled_data, m0_splits["test"], y_true_all,
            threshold=ft_thresh,
            calibrated_threshold_01=m0_calibrated_th_01
        )
        bwt_ft = round(ft_m0["f1"] - m0_baseline_perf["f1"], 4)

        step_results.append({
            "step": step_idx,
            "month": m_cur,
            "strategy": "naive_finetune",
            "current_month_perf": ft_cur,
            "month_1_perf": ft_m0,
            "backward_transfer_f1": bwt_ft
        })

        # 3. Strategy: replay
        sampled_replay = replay_buffer.get_sample()
        if len(sampled_replay) > 0:
            replay_tr = np.concatenate([cur_splits["train"], sampled_replay])
        else:
            replay_tr = cur_splits["train"]

        model_replay, rep_thresh = train_step(
            model=model_replay,
            data=step_scaled_data,
            train_indices=replay_tr,
            val_indices=cur_splits["val"],
            y_tensor=y_tensor,
            y_true_all=y_true_all,
            epochs=epochs_finetune
        )
        # Update reservoir with current month training data
        replay_buffer.add(cur_splits["train"])

        rep_cur = evaluate_on_indices(
            model_replay, step_scaled_data, cur_splits["test"], y_true_all,
            threshold=rep_thresh,
            calibrated_threshold_01=frozen_th_01
        )
        rep_m0 = evaluate_on_indices(
            model_replay, step_scaled_data, m0_splits["test"], y_true_all,
            threshold=rep_thresh,
            calibrated_threshold_01=m0_calibrated_th_01
        )
        bwt_rep = round(rep_m0["f1"] - m0_baseline_perf["f1"], 4)

        step_results.append({
            "step": step_idx,
            "month": m_cur,
            "strategy": "replay",
            "current_month_perf": rep_cur,
            "month_1_perf": rep_m0,
            "backward_transfer_f1": bwt_rep
        })

        # Update last historical validation negatives & calibrated threshold for next month
        last_val_indices = cur_splits["val"]
        last_calibrated_th_01 = calibrate_operating_threshold(
            model_scratch, step_scaled_data, cur_splits["val"], y_true_all, target_fpr=0.001
        )

    return {
        "status": "SUCCESS",
        "evaluated_months": valid_months,
        "month_0": m0,
        "month_0_baseline": m0_baseline_perf,
        "replay_reservoir_capacity": replay_reservoir_size,
        "steps": step_results
    }
