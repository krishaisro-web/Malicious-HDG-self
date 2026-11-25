"""
Malicious-HDG: Structural Adversarial Attack & Robustness Evaluation Engine.

Threat Model & Mathematical Formulation:
----------------------------------------
Implements the Multi-Instance Persistently Poisoned Target (PPT) threat model
(adapted from Nazzal et al., 2023 for heterogeneous cyber-infrastructure graphs):
1. Attacker Action Space:
   - Controls a coordinated fraction rho in {1.0, 0.5} of malicious test domains (rho=1.0 by default).
   - Injects fake resolution edges to highest-degree benign infrastructure nodes across multiple
     surfaces: IP addresses, nameservers, registrars, and TLS certificates.
   - Purity-Ranked Target Selection:
       * Infrastructure targets are ranked using TRAIN benign domains strictly (zero test exposure).
       * Purity score = degree_train_benign * (degree_train_benign / (degree_train_benign + degree_train_malicious))
       * Enforces purity >= 99% to identify bulletproof benign infrastructure (e.g., Cloudflare, Akamai).
   - Evasion Budget (k):
       * The adversary is constrained to at most k injected edges per surface (k in {1, 2, 5, 10}).
   - Optional Feature Manipulation:
       * Controllable DNS flags and certificate features (MX count, SPF/DMARC/DKIM, TTLs, cert valid len)
         are set to the median of clean training benigns.
2. Defense Evaluation (GNNGuard):
   - Computes cosine edge similarity: sim(u, v) = max(0, cos(h_u, h_v) + eps).
   - Prunes injected edges connecting disparate embeddings below tau_prune.
3. Robustness Metrics & Reviewer Convention (Metric O5):
   - Delta_att (Attack Impact): m_B - m_A (Undefended Attacked F1 - Clean F1; negative indicates degradation).
   - Residual Gap: m_C - m_A (Defended Attacked F1 - Clean F1; negative indicates gap relative to clean).
   - Defended Gain: m_C - m_B (Defended Attacked F1 - Undefended Attacked F1; positive indicates recovery).
   - Recovery Score = Defended Gain / (-Delta_att) = (m_C - m_B) / (m_A - m_B).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.metrics import evaluate_predictions
from src.hdg.data.splits import DataSplits
from src.hdg.data.features import get_feature_indices, get_primary_domain_node_type


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

    # Adhere to Constraint #6: Construct perturbations on CPU, then transfer atomically
    primary_nt = get_primary_domain_node_type(clean_data)
    orig_device = clean_data[primary_nt].y.device
    # Clone prior to transfer to prevent in-place mutation of caller's clean_data
    clean_cpu = clean_data.clone().to("cpu")
    attacked_data = clean_cpu.clone()

    rng = np.random.default_rng(seed)

    test_idx = splits.test_indices
    mal_test_idx = [idx for idx in test_idx if clean_cpu[primary_nt].y[idx].item() == 1]

    if not mal_test_idx:
        return attacked_data.to(orig_device)

    # Controlled fraction rho of malicious test domains
    n_controlled = max(1, int(round(rho * len(mal_test_idx))))
    controlled_domains = rng.choice(mal_test_idx, size=n_controlled, replace=False).tolist()

    # Identify candidate target nodes strictly from TRAIN benign domains with purity ranking
    train_benign = [idx for idx in splits.train_indices if clean_cpu[primary_nt].y[idx].item() == 0]
    train_malicious = [idx for idx in splits.train_indices if clean_cpu[primary_nt].y[idx].item() == 1]

    def get_top_train_benign_targets(edge_type: Tuple[str, str, str], min_purity: float = 0.99) -> List[int]:
        if edge_type not in clean_cpu.edge_types:
            return []
        e_arr = clean_cpu[edge_type].edge_index.numpy()
        if e_arr.shape[1] == 0:
            return []

        benign_mask = np.isin(e_arr[0], train_benign)
        mal_mask = np.isin(e_arr[0], train_malicious)

        benign_targets = e_arr[1][benign_mask]
        mal_targets = e_arr[1][mal_mask]

        if len(benign_targets) == 0:
            return []

        num_targets = int(np.max(e_arr[1])) + 1
        benign_counts = np.bincount(benign_targets, minlength=num_targets)
        mal_counts = np.bincount(mal_targets, minlength=num_targets)
        total_counts = benign_counts + mal_counts

        with np.errstate(divide="ignore", invalid="ignore"):
            purity = np.where(total_counts > 0, benign_counts / total_counts, 0.0)

        pure_mask = (purity >= min_purity) & (benign_counts > 0)
        if not np.any(pure_mask):
            pure_mask = (benign_counts > 0)

        scores = benign_counts * purity
        scores[~pure_mask] = -1.0

        ranked = np.argsort(scores)[::-1]
        valid = [int(idx) for idx in ranked if scores[idx] > 0][:20]
        return valid

    top_ips = get_top_train_benign_targets((primary_nt, "resolves_to", "ip"))
    ns_src = "apex_domain" if ("apex_domain", "uses_ns", "nameserver") in clean_cpu.edge_types else primary_nt
    reg_src = "apex_domain" if ("apex_domain", "registered_by", "registrar") in clean_cpu.edge_types else primary_nt
    top_ns = get_top_train_benign_targets((ns_src, "uses_ns", "nameserver")) if ns_src == primary_nt else []
    top_regs = get_top_train_benign_targets((reg_src, "registered_by", "registrar")) if reg_src == primary_nt else []
    top_certs = get_top_train_benign_targets((primary_nt, "uses_cert", "certificate"))

    def inject_edges(edge_type: Tuple[str, str, str], rev_type: Tuple[str, str, str], target_pool: List[int]) -> None:
        if not target_pool or edge_type not in attacked_data.edge_types:
            return
        cur_e = attacked_data[edge_type].edge_index.cpu()
        new_src = []
        new_dst = []
        for m_idx in controlled_domains:
            chosen = rng.choice(target_pool, size=min(budget, len(target_pool)), replace=False)
            for tgt in chosen:
                new_src.append(m_idx)
                new_dst.append(tgt)

        if new_src:
            new_e = torch.tensor([new_src, new_dst], dtype=torch.long, device="cpu")
            combined_e = torch.cat([cur_e, new_e], dim=1)
            unique_e = torch.unique(combined_e, dim=1)
            attacked_data[edge_type].edge_index = unique_e
            if rev_type in attacked_data.edge_types:
                attacked_data[rev_type].edge_index = torch.stack([unique_e[1], unique_e[0]], dim=0)

            dst_t = edge_type[2]
            if hasattr(attacked_data[edge_type], "edge_weight") and attacked_data[edge_type].edge_weight is not None:
                if hasattr(attacked_data[dst_t], "iif_weight") and attacked_data[dst_t].iif_weight is not None:
                    new_w = attacked_data[dst_t].iif_weight[unique_e[1]]
                    attacked_data[edge_type].edge_weight = new_w
                    if rev_type in attacked_data.edge_types:
                        attacked_data[rev_type].edge_weight = new_w.clone()
                else:
                    delattr(attacked_data[edge_type], "edge_weight")
                    if rev_type in attacked_data.edge_types and hasattr(attacked_data[rev_type], "edge_weight"):
                        delattr(attacked_data[rev_type], "edge_weight")

    # Inject edges across requested surfaces
    if "ip" in surfaces and top_ips:
        inject_edges((primary_nt, "resolves_to", "ip"), ("ip", "rev_resolves_to", primary_nt), top_ips)
    if "nameserver" in surfaces and top_ns:
        inject_edges((primary_nt, "uses_ns", "nameserver"), ("nameserver", "rev_uses_ns", primary_nt), top_ns)
    if "registrar" in surfaces and top_regs:
        inject_edges((primary_nt, "registered_by", "registrar"), ("registrar", "rev_registered_by", primary_nt), top_regs)
    if "certificate" in surfaces and top_certs:
        inject_edges((primary_nt, "uses_cert", "certificate"), ("certificate", "rev_uses_cert", primary_nt), top_certs)

    # Optional feature manipulation (resolved dynamically via named feature schema registry)
    if manipulate_features:
        x_domain = attacked_data[primary_nt].x.clone()
        controllable_indices = get_feature_indices(primary_nt, group="controllable", dim=x_domain.shape[1])
        benign_train_x = x_domain[train_benign]

        if len(train_benign) > 0 and controllable_indices:
            medians = torch.median(benign_train_x[:, controllable_indices], dim=0).values
            for m_idx in controlled_domains:
                x_domain[m_idx, controllable_indices] = medians

        attacked_data[primary_nt].x = x_domain

    return attacked_data.to(orig_device)


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
    device = next(model.parameters()).device
    primary_nt = get_primary_domain_node_type(data)
    if data[primary_nt].y.device != device:
        data = data.to(device)

    test_idx = splits.test_indices
    y_test = data[primary_nt].y[test_idx].cpu().numpy()

    with torch.no_grad():
        ew_atk = {
            et: data[et].edge_weight
            for et in data.edge_types
            if hasattr(data[et], "edge_weight") and data[et].edge_weight is not None
        } or None
        cb_atk = getattr(data[primary_nt], "char_bytes", None)
        out = model(data.x_dict, data.edge_index_dict, edge_weight_dict=ew_atk, char_bytes=cb_atk) if hasattr(model, "forward_with_aux") else model(data.x_dict, data.edge_index_dict)
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


def compute_attack_robustness_differences(
    clean_metrics: List[float],
    attacked_undefended_metrics: List[float],
    attacked_defended_metrics: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes 3 paired robustness differences with 95% bootstrap confidence intervals:
    1. Delta_att: clean - attacked_undefended (attack degradation)
    2. Residual_gap: clean - attacked_defended (residual gap under defense)
    3. Defended_gain: attacked_defended - attacked_undefended (recovery gain from defense)
    """
    c = np.array(clean_metrics, dtype=float)
    u = np.array(attacked_undefended_metrics, dtype=float)
    d = np.array(attacked_defended_metrics, dtype=float)
    n = len(c)

    delta_att = float(np.mean(u - c))
    residual_gap = float(np.mean(d - c))
    defended_gain = float(np.mean(d - u))

    rng = np.random.default_rng(seed)
    boot_delta_att = []
    boot_residual_gap = []
    boot_defended_gain = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_delta_att.append(np.mean(u[idx] - c[idx]))
        boot_residual_gap.append(np.mean(d[idx] - c[idx]))
        boot_defended_gain.append(np.mean(d[idx] - u[idx]))

    def get_ci(arr):
        return (round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4))

    return {
        "delta_att": {
            "mean": round(delta_att, 4),
            "ci_95": get_ci(boot_delta_att)
        },
        "residual_gap": {
            "mean": round(residual_gap, 4),
            "ci_95": get_ci(boot_residual_gap)
        },
        "defended_gain": {
            "mean": round(defended_gain, 4),
            "ci_95": get_ci(boot_defended_gain)
        }
    }

