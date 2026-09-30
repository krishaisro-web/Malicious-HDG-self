"""
Structural adversarial edge-injection experiment for Malicious-HDG.
Simulates an attacker who attempts evasion by adding edges from malicious domains
to popular benign infrastructure nodes (shared Nameservers, Registrars, Certificates).
Evaluates evasion rates and performance drops across budgets k in {1, 2, 5}.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.metrics import evaluate_predictions
from src.hdg.splits import DataSplits


def run_structural_attack(
    model: nn.Module,
    clean_data: HeteroData,
    splits: DataSplits,
    budgets: List[int] = [1, 2, 5],
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates GNN robustness against adversarial edge injection.
    For each budget k, adds k edges from malicious test domains to top benign infrastructure nodes.
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()
    rng = np.random.default_rng(seed)

    test_idx = splits.test_indices
    y_test = clean_data["domain"].y[test_idx].cpu().numpy()
    mal_test_idx = [idx for idx in test_idx if clean_data["domain"].y[idx].item() == 1]

    # Baseline: Clean predictions
    with torch.no_grad():
        clean_out = model(clean_data.x_dict, clean_data.edge_index_dict)
        clean_probs = F.softmax(clean_out[test_idx], dim=-1)[:, 1].cpu().numpy()

    clean_metrics = evaluate_predictions(y_test, clean_probs, threshold=0.5)

    # Identify candidate benign infrastructure targets
    # Top benign nameservers
    train_benign = [idx for idx in splits.train_indices if clean_data["domain"].y[idx].item() == 0]
    e_ns = clean_data["domain", "uses_ns", "nameserver"].edge_index.cpu().numpy()
    benign_ns_targets = []
    if e_ns.shape[1] > 0:
        mask = np.isin(e_ns[0], train_benign)
        benign_ns = e_ns[1][mask]
        if len(benign_ns) > 0:
            counts = np.bincount(benign_ns)
            benign_ns_targets = np.argsort(counts)[::-1][:10].tolist()

    attack_results: Dict[str, Any] = {
        "clean_metrics": clean_metrics,
        "budgets": {}
    }

    if not benign_ns_targets or len(mal_test_idx) == 0:
        attack_results["status"] = "No benign infrastructure targets available for injection."
        return attack_results

    # Evaluate each budget k
    for k in budgets:
        attacked_data = deepcopy(clean_data)
        cur_e_ns = attacked_data["domain", "uses_ns", "nameserver"].edge_index.cpu().numpy()

        new_src = []
        new_dst = []
        for m_idx in mal_test_idx:
            # Pick k random targets from top benign nodes
            chosen_targets = rng.choice(benign_ns_targets, size=min(k, len(benign_ns_targets)), replace=False)
            for tgt in chosen_targets:
                new_src.append(m_idx)
                new_dst.append(tgt)

        if new_src:
            aug_src = np.concatenate([cur_e_ns[0], np.array(new_src)])
            aug_dst = np.concatenate([cur_e_ns[1], np.array(new_dst)])
            attacked_edge_index = torch.tensor(np.stack([aug_src, aug_dst]), dtype=torch.long, device=device)
            attacked_data["domain", "uses_ns", "nameserver"].edge_index = attacked_edge_index
            # Also update reverse edges
            attacked_data["nameserver", "rev_uses_ns", "domain"].edge_index = torch.stack(
                [attacked_edge_index[1], attacked_edge_index[0]]
            )

        with torch.no_grad():
            atk_out = model(attacked_data.x_dict, attacked_data.edge_index_dict)
            atk_probs = F.softmax(atk_out[test_idx], dim=-1)[:, 1].cpu().numpy()

        atk_metrics = evaluate_predictions(y_test, atk_probs, threshold=0.5)

        # Compute evasion rate on malicious test samples
        clean_mal_preds = (clean_probs[y_test == 1] >= 0.5)
        atk_mal_preds = (atk_probs[y_test == 1] >= 0.5)
        # Evaded = was detected in clean, but evaded (predicted 0) under attack
        evaded_count = int(np.sum(clean_mal_preds & (~atk_mal_preds)))
        evasion_rate = float(evaded_count / max(np.sum(clean_mal_preds), 1))

        attack_results["budgets"][f"budget_{k}"] = {
            "roc_auc": atk_metrics["roc_auc"],
            "f1": atk_metrics["f1"],
            "recall": atk_metrics["recall"],
            "evaded_count": evaded_count,
            "evasion_rate": round(evasion_rate, 4),
            "auc_drop": round(clean_metrics["roc_auc"] - atk_metrics["roc_auc"], 4)
        }

    return attack_results
