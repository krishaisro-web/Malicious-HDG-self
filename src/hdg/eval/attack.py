"""
Structural adversarial edge-injection and feature-manipulation attacks for Malicious-HDG.
Implements the PPT threat model (Nazzal et al.):
- Attacker controls a coordinated fraction rho in {1.0, 0.5} of malicious test domains
- Injects fake resolution edges to highest-degree benign infrastructure (ranked using TRAIN benign domains only)
- Attacks multiple surfaces: IP, nameserver, registrar, certificate
- Optional feature manipulation: sets attacker-controllable features (TTLs, MX, SPF/DMARC/DKIM, TLS) to benign-train median
- Preserves all labels, train/val nodes, and graph structural integrity
- Evaluates at threshold fixed on clean validation and at the 0.1%-FPR operating point
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.metrics import evaluate_predictions
from src.hdg.data.splits import DataSplits


def generate_structural_attack(
    clean_data: HeteroData,
    splits: DataSplits,
    budget: int,
    rho: float = 1.0,
    surfaces: Optional[List[str]] = None,
    manipulate_features: bool = True,
    seed: int = 42
) -> HeteroData:
    """
    Constructs attacked HeteroData graph under coordinated multi-instance structural attack.
    Rankings of target benign nodes are computed STRICTLY using train benign domains to avoid leakage.
    """
    if surfaces is None:
        surfaces = ["ip", "nameserver", "registrar", "certificate"]

    rng = np.random.default_rng(seed)
    attacked_data = deepcopy(clean_data)

    test_idx = splits.test_indices
    y_test = clean_data["domain"].y[test_idx].cpu().numpy()
    mal_test_idx = [idx for idx in test_idx if clean_data["domain"].y[idx].item() == 1]

    if not mal_test_idx:
        return attacked_data

    # Controlled fraction rho of malicious test domains
    n_controlled = max(1, int(round(rho * len(mal_test_idx))))
    controlled_domains = rng.choice(mal_test_idx, size=n_controlled, replace=False).tolist()

    # Identify candidate target nodes strictly from TRAIN benign domains
    train_benign = [idx for idx in splits.train_indices if clean_data["domain"].y[idx].item() == 0]

    def get_top_train_benign_targets(edge_type: Tuple[str, str, str]) -> List[int]:
        if edge_type not in clean_data.edge_types:
            return []
        e_arr = clean_data[edge_type].edge_index.cpu().numpy()
        if e_arr.shape[1] == 0:
            return []
        mask = np.isin(e_arr[0], train_benign)
        targets = e_arr[1][mask]
        if len(targets) == 0:
            return []
        counts = np.bincount(targets)
        return np.argsort(counts)[::-1][:20].tolist()

    top_ips = get_top_train_benign_targets(("domain", "resolves_to", "ip"))
    top_ns = get_top_train_benign_targets(("domain", "uses_ns", "nameserver"))
    top_regs = get_top_train_benign_targets(("domain", "registered_by", "registrar"))
    top_certs = get_top_train_benign_targets(("domain", "uses_cert", "certificate"))

    def inject_edges(edge_type: Tuple[str, str, str], rev_type: Tuple[str, str, str], target_pool: List[int]) -> None:
        if not target_pool or edge_type not in attacked_data.edge_types:
            return
        cur_e = attacked_data[edge_type].edge_index.cpu().numpy()
        new_src = []
        new_dst = []
        for m_idx in controlled_domains:
            chosen = rng.choice(target_pool, size=min(budget, len(target_pool)), replace=False)
            for tgt in chosen:
                new_src.append(m_idx)
                new_dst.append(tgt)

        if new_src:
            all_src = np.concatenate([cur_e[0], np.array(new_src)])
            all_dst = np.concatenate([cur_e[1], np.array(new_dst)])
            # Deduplicate pairs
            pairs = list(set(zip(all_src.tolist(), all_dst.tolist())))
            pairs.sort()
            s_t = torch.tensor([p[0] for p in pairs], dtype=torch.long)
            d_t = torch.tensor([p[1] for p in pairs], dtype=torch.long)
            new_idx = torch.stack([s_t, d_t])
            attacked_data[edge_type].edge_index = new_idx
            if rev_type in attacked_data.edge_types:
                attacked_data[rev_type].edge_index = torch.stack([d_t, s_t])

    # Inject edges across requested surfaces
    if "ip" in surfaces and top_ips:
        inject_edges(("domain", "resolves_to", "ip"), ("ip", "rev_resolves_to", "domain"), top_ips)
    if "nameserver" in surfaces and top_ns:
        inject_edges(("domain", "uses_ns", "nameserver"), ("nameserver", "rev_uses_ns", "domain"), top_ns)
    if "registrar" in surfaces and top_regs:
        inject_edges(("domain", "registered_by", "registrar"), ("registrar", "rev_registered_by", "domain"), top_regs)
    if "certificate" in surfaces and top_certs:
        inject_edges(("domain", "uses_cert", "certificate"), ("certificate", "rev_uses_cert", "domain"), top_certs)

    # Optional feature manipulation
    if manipulate_features:
        # Controllable feature indices:
        # mx_count (3), has_spf (4), has_dmarc (5), has_dkim (6), ttl_a (8), ttl_ns (9), tls_cert_count (10), cert_valid_len (11)
        controllable_indices = [3, 4, 5, 6, 8, 9, 10, 11]
        x_domain = attacked_data["domain"].x.clone()
        benign_train_x = x_domain[train_benign]

        if len(train_benign) > 0:
            medians = torch.median(benign_train_x[:, controllable_indices], dim=0).values
            for m_idx in controlled_domains:
                x_domain[m_idx, controllable_indices] = medians

        attacked_data["domain"].x = x_domain

    return attacked_data


def evaluate_attacked_model(
    model: nn.Module,
    data: HeteroData,
    splits: DataSplits,
    threshold: float,
    operating_threshold_01: float,
    clean_detected_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Evaluates model predictions on attacked graph at clean validation thresholds."""
    model.eval()
    test_idx = splits.test_indices
    y_test = data["domain"].y[test_idx].cpu().numpy()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out[test_idx], dim=-1)[:, 1].cpu().numpy()

    eval_res = evaluate_predictions(y_test, probs, threshold=threshold)
    test_negs = probs[y_test == 0]
    test_pos = probs[y_test == 1]

    # Realised metrics at operating threshold (0.1% FPR)
    tpr_01 = float(np.mean(test_pos >= operating_threshold_01)) if len(test_pos) > 0 else 0.0
    realised_fpr_01 = float(np.mean(test_negs >= operating_threshold_01)) if len(test_negs) > 0 else 0.0

    # Evasion rate on clean-detected malicious domains
    evasion_rate = 0.0
    evaded_count = 0
    if clean_detected_mask is not None and len(clean_detected_mask) > 0:
        atk_detected = (probs[y_test == 1] >= threshold)
        evaded_mask = clean_detected_mask & (~atk_detected)
        evaded_count = int(np.sum(evaded_mask))
        evasion_rate = float(evaded_count / max(np.sum(clean_detected_mask), 1))

    return {
        "roc_auc": eval_res["roc_auc"],
        "pr_auc": eval_res["pr_auc"],
        "f1": eval_res["f1"],
        "recall": eval_res["recall"],
        "precision": eval_res["precision"],
        "tpr_at_0.1pct_fpr": round(tpr_01, 4),
        "realised_fpr_0.1pct": round(realised_fpr_01, 6),
        "evaded_count": evaded_count,
        "evasion_rate": round(evasion_rate, 4),
        "y_prob": probs
    }
