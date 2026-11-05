"""
Post-Hoc Stacker with Guilt-By-Association (GBA) Features for Malicious-HDG.
Implements:
1. Smoothed Guilt-By-Association (GBA) node scoring:
   p_v = (m_v + a * pi) / (n_v + a)
   for a in [5, 10], where n_v is connected training domains, m_v is malicious training domains,
   and pi is training malicious base rate.
2. Feature extraction: GBA features per infrastructure surface (IP, NS, Registrar, ASN, Cert)
   aggregated (mean, max) alongside base model probabilities:
   - HeteroGNN (SAGE)
   - HeteroGNN (ATTN)
   - Tabular XGBoost
   - Lexical TF-IDF LogReg
3. Validation-only 5-fold cross-fitting of meta-learner (LogisticRegression / Ridge)
   to ensure strictly leak-free meta-training without touching test sets.
4. Pre-committed Stop Rule evaluation: requires >= +3.0 percentage points TPR @ 0.1% FPR
   over SAGE with 95% bootstrap CI excluding 0; otherwise transparently reports negative result.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData

from src.hdg.metrics import (
    compute_bootstrap_cis,
    compute_tpr_at_fpr,
    evaluate_predictions,
    paired_bootstrap_auc_difference
)


def compute_smoothed_gba_scores(
    data: HeteroData,
    train_domain_indices: np.ndarray,
    train_labels: np.ndarray,
    smoothing_a: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Computes smoothed Guilt-By-Association (GBA) maliciousness scores for all infrastructure entities.
    p_v = (m_v + a * pi) / (n_v + a)
    Strictly uses training domains only.
    """
    pi = float(np.mean(train_labels)) if len(train_labels) > 0 else 0.3325
    train_domain_set = set(train_domain_indices.tolist())
    train_label_dict = dict(zip(train_domain_indices.tolist(), train_labels.tolist()))

    gba_scores: Dict[str, np.ndarray] = {}

    for nt in data.node_types:
        if nt == "domain":
            continue
        num_nodes = data[nt].num_nodes
        n_v = np.zeros(num_nodes, dtype=np.float64)
        m_v = np.zeros(num_nodes, dtype=np.float64)

        # Inspect edges connecting domain to this node type
        for et in data.edge_types:
            src_t, rel, dst_t = et
            if src_t == "domain" and dst_t == nt:
                e_arr = data[et].edge_index.cpu().numpy()
                for d_idx, infra_idx in zip(e_arr[0], e_arr[1]):
                    if d_idx in train_domain_set:
                        n_v[infra_idx] += 1.0
                        if train_label_dict.get(d_idx, 0) == 1:
                            m_v[infra_idx] += 1.0
            elif src_t == nt and dst_t == "domain":
                e_arr = data[et].edge_index.cpu().numpy()
                for infra_idx, d_idx in zip(e_arr[0], e_arr[1]):
                    if d_idx in train_domain_set:
                        n_v[infra_idx] += 1.0
                        if train_label_dict.get(d_idx, 0) == 1:
                            m_v[infra_idx] += 1.0

        p_v = (m_v + smoothing_a * pi) / (n_v + smoothing_a)
        gba_scores[nt] = p_v

    return gba_scores


def extract_gba_features(
    data: HeteroData,
    target_domain_indices: np.ndarray,
    gba_scores: Dict[str, np.ndarray],
    default_prior: float = 0.3325
) -> np.ndarray:
    """
    Extracts mean and max GBA features across connected entities for each domain.
    Returns: feature matrix of shape (len(target_domain_indices), 2 * num_entity_types).
    """
    entity_types = [nt for nt in data.node_types if nt != "domain"]
    n_domains = len(target_domain_indices)
    n_features = len(entity_types) * 2
    features = np.full((n_domains, n_features), default_prior, dtype=np.float64)

    # Build domain -> connected entities map
    domain_to_entities: Dict[int, Dict[str, List[int]]] = {
        d_idx: {nt: [] for nt in entity_types} for d_idx in target_domain_indices
    }
    target_set = set(target_domain_indices.tolist())

    for et in data.edge_types:
        src_t, rel, dst_t = et
        if src_t == "domain" and dst_t in entity_types:
            e_arr = data[et].edge_index.cpu().numpy()
            for d_idx, infra_idx in zip(e_arr[0], e_arr[1]):
                if d_idx in target_set:
                    domain_to_entities[d_idx][dst_t].append(infra_idx)
        elif src_t in entity_types and dst_t == "domain":
            e_arr = data[et].edge_index.cpu().numpy()
            for infra_idx, d_idx in zip(e_arr[0], e_arr[1]):
                if d_idx in target_set:
                    domain_to_entities[d_idx][src_t].append(infra_idx)

    for i, d_idx in enumerate(target_domain_indices):
        col = 0
        for nt in entity_types:
            connected = domain_to_entities[d_idx][nt]
            if connected and nt in gba_scores:
                scores = gba_scores[nt][connected]
                features[i, col] = float(np.mean(scores))
                features[i, col + 1] = float(np.max(scores))
            else:
                features[i, col] = default_prior
                features[i, col + 1] = default_prior
            col += 2

    return features


class PostHocStacker:
    """
    Cross-fitted post-hoc stacker combining GNNs, tree baselines, lexical models,
    and smoothed GBA topological indicators.
    """

    def __init__(self, c_reg: float = 1.0, random_state: int = 42):
        self.c_reg = c_reg
        self.random_state = random_state
        self.meta_learner = LogisticRegression(
            C=self.c_reg,
            random_state=self.random_state,
            max_iter=500
        )
        self.is_fitted = False

    def fit_cross_validated(
        self,
        base_probs: np.ndarray,
        gba_features: np.ndarray,
        y_val: np.ndarray,
        n_splits: int = 5
    ) -> None:
        """
        Fits stacker meta-learner strictly on validation data using 5-fold cross-fitting.
        base_probs: (N_val, num_base_models)
        gba_features: (N_val, num_gba_features)
        y_val: (N_val,)
        """
        x_val = np.hstack([base_probs, gba_features])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Cross-fitted out-of-fold training
        oof_preds = np.zeros(len(y_val))
        for train_idx, fit_idx in skf.split(x_val, y_val):
            fold_model = LogisticRegression(C=self.c_reg, random_state=self.random_state, max_iter=500)
            fold_model.fit(x_val[train_idx], y_val[train_idx])
            oof_preds[fit_idx] = fold_model.predict_proba(x_val[fit_idx])[:, 1]

        # Fit final meta-learner on complete validation set
        self.meta_learner.fit(x_val, y_val)
        self.is_fitted = True

    def predict_proba(self, base_probs: np.ndarray, gba_features: np.ndarray) -> np.ndarray:
        """Generates stacked ensemble malicious probability scores."""
        if not self.is_fitted:
            raise RuntimeError("PostHocStacker must be fitted before predict_proba.")
        x = np.hstack([base_probs, gba_features])
        return self.meta_learner.predict_proba(x)[:, 1]


def evaluate_stacker_against_stop_rule(
    y_test: np.ndarray,
    sage_probs: np.ndarray,
    stacker_probs: np.ndarray,
    y_val: np.ndarray,
    sage_val_probs: np.ndarray,
    stacker_val_probs: np.ndarray,
    stop_rule_gain_threshold: float = 0.030,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates stacked predictions vs baseline SAGE against pre-committed stop rule.
    Stop rule criterion:
    - Delta TPR @ 0.1% FPR >= +3.0 percentage points (+0.030)
    - 95% bootstrap CI strictly excludes zero
    If not met, declares transparent negative result.
    """
    # 1. Base SAGE operating point
    tpr_sage, th_s, _, _ = compute_tpr_at_fpr(y_test, sage_probs, y_val, sage_val_probs, target_fpr=0.001)
    
    # 2. Stacker operating point
    tpr_stack, th_st, _, _ = compute_tpr_at_fpr(y_test, stacker_probs, y_val, stacker_val_probs, target_fpr=0.001)

    delta_tpr = tpr_stack - tpr_sage

    # 3. Bootstrap CI on Delta TPR
    rng = np.random.default_rng(seed)
    n = len(y_test)
    delta_boots = []
    test_negs_mask = (y_test == 0)
    test_pos_mask = (y_test == 1)

    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        yb = y_test[idx]
        if np.sum(yb == 1) == 0 or np.sum(yb == 0) == 0:
            continue
        tpr_s_b = float(np.mean(sage_probs[idx][yb == 1] >= th_s))
        tpr_st_b = float(np.mean(stacker_probs[idx][yb == 1] >= th_st))
        delta_boots.append(tpr_st_b - tpr_s_b)

    ci_lower = float(np.percentile(delta_boots, 2.5)) if delta_boots else 0.0
    ci_upper = float(np.percentile(delta_boots, 97.5)) if delta_boots else 0.0

    passed_threshold = (delta_tpr >= stop_rule_gain_threshold)
    ci_excludes_zero = (ci_lower > 0.0)
    is_positive_result = (passed_threshold and ci_excludes_zero)

    status = "POSITIVE_RESULT" if is_positive_result else "NEGATIVE_RESULT (PRE-COMMITTED)"
    interpretation = (
        f"Stacker achieved {tpr_stack:.2%} TPR vs SAGE {tpr_sage:.2%} TPR (Δ: {delta_tpr:+.2%}, "
        f"95% CI: [{ci_lower:+.2%}, {ci_upper:+.2%}]). "
    )
    if is_positive_result:
        interpretation += "Surpassed pre-committed threshold (>= +3.0 pp, CI > 0)."
    else:
        interpretation += (
            "Did not surpass pre-committed +3.0 pp margin or CI includes 0. "
            "Reported transparently as a negative result; the core thesis contribution "
            "remains the FPR-anchored evaluation methodology and leakage audit framework."
        )

    return {
        "status": status,
        "is_positive_result": is_positive_result,
        "tpr_sage_at_01pct_fpr": round(tpr_sage, 4),
        "tpr_stacker_at_01pct_fpr": round(tpr_stack, 4),
        "delta_tpr": round(delta_tpr, 4),
        "bootstrap_ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
        "interpretation": interpretation
    }
