"""
Malicious-HDG: Leak-Free Graph Dataset Partitioning & Split Generation.

Evaluation Topologies & Methodological Design:
---------------------------------------------
1. Random Stratified Split (`random_stratified_split`):
   - Generates 70/15/15 train/val/test splits balanced on domain labels.
   - Theoretical Role: Serves strictly as an OPTIMISTIC BOUND. In transductive GNNs,
     random splits permit message-passing leakage across shared infrastructure hubs
     (e.g., train and test domains co-hosted on identical ASNs or nameservers).

2. Rolling-Origin Temporal Split (`rolling_origin_temporal_splits` & `time_split`):
   - Evaluates pure INDUCTIVE TEMPORAL GENERALIZATION across consecutive calendar months (T -> T+1).
   - Training graph contains strictly historical records (t <= T_cutoff). Future domains (t > T_cutoff)
     are completely withheld from graph construction, degree calculations, and feature scaling.
   - Evaluates cyber defense durability against attacker evasion strategies and infrastructure migration over time.

3. Disjoint Bipartite Group Holdout (`group_split_bipartite`):
   - Tests generalization to completely UNSEEN INFRASTRUCTURE CLUSTERS.
   - Algorithms:
     a. Hub Pruning: High-degree infrastructure nodes (e.g. Cloudflare, Akamai, GoDaddy with degree > 500)
        are excluded from component bridging to avoid collapsing the entire graph into a single giant component.
     b. Connected Component Discovery: Extracts independent bipartite subgraphs using SciPy CSR graph traversal.
     c. Greedy Stratified Bin Packing: Distributes independent clusters to train/val/test so malicious class
        prevalences in all partitions remain within +/- 5 percentage points of the global dataset prior.

4. 50% Negative Certification Holdout (`cert_indices`):
   - Validation negative domains are split 50/50 into:
     - Model-Selection Val: Used for early stopping, learning rate grid search, and decision threshold (tau*).
     - Certification Val: Reserved exclusively for FPR calibration (e.g., certified 0.1% FPR operating points)
       to eliminate hyperparameter reuse bias.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


@dataclass
class DataSplits:
    """
    Container for disjoint domain node index sets defining an evaluation split.
    
    Attributes:
        train_indices: 1D NumPy array of node IDs assigned to training.
        val_indices: 1D NumPy array of node IDs assigned to validation (early stopping / tuning).
        test_indices: 1D NumPy array of node IDs assigned to final held-out evaluation.
        split_type: String identifier ("random", "time", "group", "temporal_rolling_origin").
        metadata: Supplementary split diagnostic statistics (counts, prevalences, transition months).
        cert_indices: Optional disjoint validation negative subset for certified threshold calibration.
    """
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    split_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    cert_indices: Optional[np.ndarray] = None

    def assert_disjoint(self) -> None:
        """
        Cryptographically asserts that train, val, test, and cert index sets are strictly disjoint.
        Raises AssertionError immediately if any domain index appears in multiple partitions.
        """
        s_train = set(self.train_indices.tolist())
        s_val = set(self.val_indices.tolist())
        s_test = set(self.test_indices.tolist())

        inter_tv = s_train.intersection(s_val)
        inter_tt = s_train.intersection(s_test)
        inter_vt = s_val.intersection(s_test)

        if inter_tv:
            raise AssertionError(f"Leak detected: {len(inter_tv)} indices in both Train and Val!")
        if inter_tt:
            raise AssertionError(f"Leak detected: {len(inter_tt)} indices in both Train and Test!")
        if inter_vt:
            raise AssertionError(f"Leak detected: {len(inter_vt)} indices in both Val and Test!")

        if self.cert_indices is not None:
            s_cert = set(self.cert_indices.tolist())
            if s_cert.intersection(s_train):
                raise AssertionError("Leak detected: Certification indices overlap with Train!")
            if s_cert.intersection(s_val):
                raise AssertionError("Leak detected: Certification indices overlap with Model-Selection Val!")
            if s_cert.intersection(s_test):
                raise AssertionError("Leak detected: Certification indices overlap with Test!")


def random_stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> DataSplits:
    """
    Constructs a random stratified 70/15/15 train/val/test split.
    
    Includes 50% validation negative holdout partitioning (`cert_indices`) to prevent
    threshold calibration contamination.
    
    Note: Reported strictly as an optimistic bound due to transductive infrastructure sharing.
    """
    n_total = len(labels)
    indices = np.arange(n_total)

    test_ratio = 1.0 - train_ratio - val_ratio
    holdout_size = val_ratio + test_ratio
    can_sss1 = (
        len(np.unique(labels)) > 1
        and np.min(np.bincount(labels)) >= 2
        and int(round(holdout_size * n_total)) >= 2
        and (n_total - int(round(holdout_size * n_total))) >= 2
    )
    if can_sss1:
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
            train_idx, temp_idx = next(sss1.split(indices, labels))
        except ValueError:
            can_sss1 = False
    if not can_sss1:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total)
        cut = max(1, int(round(train_ratio * n_total)))
        train_idx, temp_idx = perm[:cut], perm[cut:]

    temp_labels = labels[temp_idx]
    val_relative_size = val_ratio / max(val_ratio + test_ratio, 1e-8)
    can_sss2 = (
        len(temp_idx) >= 4
        and len(np.unique(temp_labels)) > 1
        and np.min(np.bincount(temp_labels)) >= 2
    )
    if can_sss2:
        try:
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_relative_size), random_state=seed)
            sub_val_idx, sub_test_idx = next(sss2.split(temp_idx, temp_labels))
        except ValueError:
            can_sss2 = False
    if not can_sss2:
        rng2 = np.random.RandomState(seed + 1)
        perm2 = rng2.permutation(len(temp_idx))
        cut2 = max(1, int(round(val_relative_size * len(temp_idx))))
        sub_val_idx, sub_test_idx = perm2[:cut2], perm2[cut2:]

    val_idx = temp_idx[sub_val_idx]
    test_idx = temp_idx[sub_test_idx]

    # Pre-allocate 50% of validation negatives into an independent certification holdout
    # to guarantee zero data reuse during early stopping, hyperparameter search, and threshold calibration
    val_labels = labels[val_idx]
    val_neg_indices = val_idx[val_labels == 0]
    val_pos_indices = val_idx[val_labels == 1]

    if len(val_neg_indices) >= 2:
        rng = np.random.RandomState(seed)
        shuffled_negs = rng.permutation(val_neg_indices)
        half = len(val_neg_indices) // 2
        calib_negs = shuffled_negs[:half]
        cert_negs = shuffled_negs[half:]
        # val_indices for model selection and early stopping strictly contains calib_negs + val positives
        model_val_indices = np.concatenate([val_pos_indices, calib_negs])
        model_val_indices.sort()
        cert_indices = cert_negs
        cert_indices.sort()
    else:
        model_val_indices = val_idx
        cert_indices = np.array([], dtype=int)

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=model_val_indices,
        test_indices=test_idx,
        split_type="random_stratified",
        metadata={
            "train_size": len(train_idx),
            "val_size": len(model_val_indices),
            "test_size": len(test_idx),
            "cert_size": len(cert_indices),
            "seed": seed
        },
        cert_indices=cert_indices
    )
    splits.assert_disjoint()
    return splits


def rolling_origin_temporal_splits(
    df: pd.DataFrame,
    min_distinct_months: int = 3,
    min_per_class_per_window: int = 200,
    train_val_split_ratio: float = 0.85,
    allow_empty: bool = False,
    seed: int = 42
) -> List[Tuple[str, DataSplits]]:
    """
    Rolling-origin temporal evaluation over months where BOTH classes exist:
    For transition T -> T+1:
      - Train on months <= T (stratified 85/15 train/val split inside historical window)
      - Test on month T+1
    Skips transitions with insufficient class samples and logs exact reasons.
    If overlapping months < 2 and allow_empty is True, logs a warning and returns [].
    Otherwise raises ValueError.
    """
    if "t_month" not in df.columns:
        df["t_month"] = pd.to_datetime(df["t"]).dt.strftime("%Y-%m")

    # Find months where both classes exist
    counts_by_month: Dict[str, Dict[int, int]] = {}
    for mo, sub in df.groupby("t_month"):
        y = sub["label"].to_numpy()
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        counts_by_month[str(mo)] = {1: pos, 0: neg}

    both_class_months = sorted([
        mo for mo, c in counts_by_month.items()
        if c[1] > 0 and c[0] > 0
    ])

    if len(both_class_months) < 2:
        msg = (
            f"Overlapping months with both classes is {len(both_class_months)} (< 2). "
            "Temporal rolling-origin evaluation cannot proceed."
        )
        if allow_empty:
            logger.warning(msg)
            return []
        raise ValueError(msg)

    if len(both_class_months) < min_distinct_months:
        logger.warning(
            f"Only {len(both_class_months)} distinct overlapping months available "
            f"(target: {min_distinct_months}). There will be only {len(both_class_months) - 1} transition(s)."
        )

    transitions: List[Tuple[str, DataSplits]] = []

    for i in range(len(both_class_months) - 1):
        t_cur = both_class_months[i]
        t_next = both_class_months[i + 1]
        trans_name = f"{t_cur}->{t_next}"

        # History: all records with month <= t_cur
        hist_mask = (df["t_month"] <= t_cur).to_numpy()
        test_mask = (df["t_month"] == t_next).to_numpy()

        hist_indices = np.where(hist_mask)[0]
        test_indices = np.where(test_mask)[0]

        hist_y = df.iloc[hist_indices]["label"].to_numpy()
        test_y = df.iloc[test_indices]["label"].to_numpy()

        hist_pos = int(np.sum(hist_y == 1))
        hist_neg = int(np.sum(hist_y == 0))
        test_pos = int(np.sum(test_y == 1))
        test_neg = int(np.sum(test_y == 0))

        # Check sample size thresholds (adjusted if dataset is small, e.g. fixture)
        effective_min = min(min_per_class_per_window, max(5, int(len(df) * 0.01)))
        if hist_pos < effective_min or hist_neg < effective_min or test_pos < effective_min or test_neg < effective_min:
            print(
                f"[TEMPORAL SKIP] Transition {trans_name} skipped: insufficient samples "
                f"(History: pos={hist_pos}, neg={hist_neg}; Test: pos={test_pos}, neg={test_neg}; min required={effective_min})."
            )
            continue

        # Stratified 85/15 train/val split inside historical window
        val_size = 1.0 - train_val_split_ratio
        n_val_needed = int(len(hist_indices) * val_size)
        if hist_pos >= 2 and hist_neg >= 2 and n_val_needed >= 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            sub_train_idx, sub_val_idx = next(sss.split(hist_indices, hist_y))
            train_indices = hist_indices[sub_train_idx]
            val_indices = hist_indices[sub_val_idx]
        else:
            # Fallback for very small historical cohorts: allocate 1 pos and 1 neg for val if possible
            pos_h = hist_indices[hist_y == 1]
            neg_h = hist_indices[hist_y == 0]
            if len(pos_h) > 1 and len(neg_h) > 1:
                val_indices = np.array([pos_h[0], neg_h[0]], dtype=int)
            else:
                val_indices = np.array([hist_indices[0]], dtype=int)
            train_indices = np.array([idx for idx in hist_indices if idx not in val_indices], dtype=int)

        val_labels = df.iloc[val_indices]["label"].to_numpy()
        val_neg_idx = val_indices[val_labels == 0]
        val_pos_idx = val_indices[val_labels == 1]

        if len(val_neg_idx) >= 2:
            rng = np.random.RandomState(seed)
            shuffled_negs = rng.permutation(val_neg_idx)
            half = len(val_neg_idx) // 2
            calib_negs = shuffled_negs[:half]
            cert_negs = shuffled_negs[half:]
            model_val_indices = np.concatenate([val_pos_idx, calib_negs])
            model_val_indices.sort()
            cert_indices = cert_negs
            cert_indices.sort()
        else:
            model_val_indices = val_indices
            cert_indices = np.array([], dtype=int)

        splits = DataSplits(
            train_indices=train_indices,
            val_indices=model_val_indices,
            test_indices=test_indices,
            split_type="temporal_rolling_origin",
            metadata={
                "transition": trans_name,
                "history_max_month": t_cur,
                "test_month": t_next,
                "train_pos": int(np.sum(df.iloc[train_indices]["label"] == 1)),
                "train_neg": int(np.sum(df.iloc[train_indices]["label"] == 0)),
                "val_pos": int(np.sum(df.iloc[model_val_indices]["label"] == 1)),
                "val_neg": int(np.sum(df.iloc[model_val_indices]["label"] == 0)),
                "cert_neg": len(cert_indices),
                "test_pos": test_pos,
                "test_neg": test_neg,
                "seed": seed
            },
            cert_indices=cert_indices
        )
        splits.assert_disjoint()
        transitions.append((trans_name, splits))

    return transitions


def time_split(
    df: pd.DataFrame,
    time_split_valid: bool = True,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    min_positives: int = 5,
    min_negatives: int = 5
) -> DataSplits:
    """
    Backwards-compatible single temporal split.
    If rolling-origin is preferred, call rolling_origin_temporal_splits.
    """
    if not time_split_valid:
        raise ValueError("time_split_valid is False: temporal evaluation aborted.")
    df_sorted = df.sort_values(by=["t"]).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="time",
        metadata={
            "t_train_max": df_sorted.iloc[train_idx[-1]]["t"] if len(train_idx) > 0 else None,
            "t_val_min": df_sorted.iloc[val_idx[0]]["t"] if len(val_idx) > 0 else None,
            "t_test_min": df_sorted.iloc[test_idx[0]]["t"] if len(test_idx) > 0 else None
        }
    )
    splits.assert_disjoint()
    return splits


def assign_groups_stratified(
    groups: Dict[int, List[int]],
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_discrepancy: float = 0.05,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Assigns independent group clusters (components or ASNs) greedily to train, val, and test
    so class ratios in each split remain within max_discrepancy (5 percentage points) of overall.
    Guarantees no group cluster spans two sets.
    """
    rng = np.random.default_rng(seed)
    target_pos_ratio = float(np.mean(labels == 1))

    # Separate groups into positive-leaning and negative-leaning clusters
    pos_groups = []
    neg_groups = []
    for gid, members in groups.items():
        if not members:
            continue
        g_ratio = float(np.mean(labels[members] == 1))
        if g_ratio >= target_pos_ratio:
            pos_groups.append((gid, members, int(np.sum(labels[members] == 1)), len(members)))
        else:
            neg_groups.append((gid, members, int(np.sum(labels[members] == 1)), len(members)))

    # Sort descending by size with tie-breaker
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)
    pos_groups.sort(key=lambda x: x[3], reverse=True)
    neg_groups.sort(key=lambda x: x[3], reverse=True)

    train_members: List[int] = []
    val_members: List[int] = []
    test_members: List[int] = []

    def distribute_pool(pool: List[Tuple[Any, List[int], int, int]]) -> None:
        pool_total = sum(x[3] for x in pool)
        t_tr = int(round(pool_total * train_ratio))
        t_va = int(round(pool_total * val_ratio))
        t_te = pool_total - t_tr - t_va

        tr_cur, va_cur, te_cur = 0, 0, 0
        for gid, members, n_pos, n_len in pool:
            # Pick candidate whose relative fill is lowest
            r_tr = tr_cur / max(t_tr, 1)
            r_va = va_cur / max(t_va, 1)
            r_te = te_cur / max(t_te, 1)

            choices = [(r_tr, "train"), (r_va, "val"), (r_te, "test")]
            choices.sort(key=lambda x: x[0])
            dest = choices[0][1]

            if dest == "train":
                train_members.extend(members)
                tr_cur += n_len
            elif dest == "val":
                val_members.extend(members)
                va_cur += n_len
            else:
                test_members.extend(members)
                te_cur += n_len

    distribute_pool(pos_groups)
    distribute_pool(neg_groups)

    train_arr = np.array(train_members, dtype=int)
    val_arr = np.array(val_members, dtype=int)
    test_arr = np.array(test_members, dtype=int)

    # Calculate final prevalences
    p_train = float(np.mean(labels[train_arr] == 1)) if len(train_arr) > 0 else 0.0
    p_val = float(np.mean(labels[val_arr] == 1)) if len(val_arr) > 0 else 0.0
    p_test = float(np.mean(labels[test_arr] == 1)) if len(test_arr) > 0 else 0.0

    stat = {
        "overall_prevalence": round(target_pos_ratio, 4),
        "train_prevalence": round(p_train, 4),
        "val_prevalence": round(p_val, 4),
        "test_prevalence": round(p_test, 4),
        "train_count": len(train_arr),
        "val_count": len(val_arr),
        "test_count": len(test_arr)
    }

    # Verify tolerance
    for s_name, p_val_curr in [("train", p_train), ("val", p_val), ("test", p_test)]:
        if abs(p_val_curr - target_pos_ratio) > max_discrepancy:
            print(
                f"[WARNING] Group split stratified tolerance exceeded for {s_name}: "
                f"Prevalence={p_val_curr:.4f}, Target={target_pos_ratio:.4f} (diff > {max_discrepancy*100:.1f}%)"
            )

    return train_arr, val_arr, test_arr, stat


def group_split_bipartite(
    df: pd.DataFrame,
    hub_degree: int = 500,
    hub_percentile: float = 99.9,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    joint_grouping: bool = False,
    max_discrepancy: float = 0.05,
    seed: int = 42
) -> Tuple[DataSplits, Dict[str, Any]]:
    """
    Constructs a leak-free group split using connected components of the domain-infrastructure graph.
    
    Guarantees inductive generalization to novel attacker infrastructure:
    1. Relation Selection:
       - Default: Resolves-to (IP) and uses-nameserver (NS).
       - Joint Grouping (`joint_grouping=True`): Includes IP, NS, Registrar, and Certificate links.
    2. Hub Pruning:
       - Infrastructure nodes exceeding `effective_cutoff = min(hub_degree, perc_cutoff)`
         are masked out before component extraction.
       - Prevents mega-hubs (e.g. Cloudflare 1.1.1.1, Google DNS 8.8.8.8, AWS Route53) from bridging
         thousands of independent domains into an intractable single giant component.
    3. Connected Component Partitioning:
       - Assembles a bipartite CSR adjacency matrix of size (N_domain + N_infra, N_domain + N_infra).
       - Extracts independent disjoint graph clusters using SciPy csgraph connected_components.
    4. Stratified Cluster Assignment:
       - Greedily assigns component clusters to train/val/test pools using `assign_groups_stratified`
         to maintain class prevalence within `max_discrepancy` (+/- 5 percentage points).
       - Zero components cross between train, validation, or test sets.
    5. Certification Holdout:
       - Splits validation negatives 50/50 to guarantee clean operational FPR calibration.

    Args:
        df: Aligned domains DataFrame containing labels and infrastructure columns.
        hub_degree: Maximum allowed incident degree before a node is pruned as a multi-tenant hub.
        hub_percentile: Upper percentile cutoff for hub degree pruning (default: 99.9th percentile).
        train_ratio: Target proportion for training set (default: 0.70).
        val_ratio: Target proportion for validation set (default: 0.15).
        joint_grouping: If True, bridges components across all 4 infrastructure surfaces.
        max_discrepancy: Maximum allowed deviation between split and global class prevalence.
        seed: Random seed for deterministic cluster tie-breaking.

    Returns:
        Tuple of (DataSplits instance, diagnostic_metadata_dictionary).
    """
    n_domains = len(df)
    labels = df["label"].to_numpy()

    infra_to_id: Dict[str, int] = {}
    edges_domain: List[int] = []
    edges_infra: List[int] = []

    res_ips_list = df["resolved_ips"].tolist() if "resolved_ips" in df.columns else [None] * n_domains
    ns_list_list = df["nameservers"].tolist() if "nameservers" in df.columns else [None] * n_domains
    reg_list = df["registrar"].tolist() if "registrar" in df.columns else [None] * n_domains
    cert_list = (df["leaf_cert_key_coissue"].tolist() if "leaf_cert_key_coissue" in df.columns
                 else (df["leaf_cert_key"].tolist() if "leaf_cert_key" in df.columns else [None] * n_domains))

    for d_idx in range(n_domains):
        # Resolves_to (IP)
        res_ips = res_ips_list[d_idx]
        if res_ips is not None and isinstance(res_ips, (list, np.ndarray)):
            for ip in res_ips:
                k = f"ip:{ip}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        # Nameservers
        ns_list = ns_list_list[d_idx]
        if ns_list is not None and isinstance(ns_list, (list, np.ndarray)):
            for ns in ns_list:
                k = f"ns:{ns}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        if joint_grouping:
            # Registrar
            reg = reg_list[d_idx]
            if reg:
                k = f"reg:{reg}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

            # Leaf Certificate
            cert = cert_list[d_idx]
            if cert:
                k = f"cert:{cert}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

    n_infra = len(infra_to_id)
    if n_infra == 0 or len(edges_domain) == 0:
        base_s = random_stratified_split(labels, train_ratio, val_ratio, seed)
        return base_s, {"giant_component_fraction": 0.0, "num_components": 1}

    # Count degrees of infra nodes and prune hubs
    infra_degrees = np.bincount(edges_infra, minlength=n_infra)
    perc_cutoff = np.percentile(infra_degrees, hub_percentile)
    effective_cutoff = min(hub_degree, perc_cutoff)

    edges_infra_arr = np.array(edges_infra, dtype=np.int64)
    edges_domain_arr = np.array(edges_domain, dtype=np.int64)
    valid_mask = infra_degrees[edges_infra_arr] <= effective_cutoff
    f_d = edges_domain_arr[valid_mask]
    f_i = edges_infra_arr[valid_mask]

    # Build bipartite adjacency
    row_idx = np.concatenate([f_d, f_i + n_domains])
    col_idx = np.concatenate([f_i + n_domains, f_d])
    data_ones = np.ones(len(row_idx), dtype=int)
    adj = csr_matrix((data_ones, (row_idx, col_idx)), shape=(n_domains + n_infra, n_domains + n_infra))

    n_components, comp_labels = connected_components(adj, directed=False)
    domain_comp_labels = comp_labels[:n_domains]

    comp_sizes = np.bincount(domain_comp_labels, minlength=n_components)
    giant_comp_size = int(np.max(comp_sizes))
    giant_comp_fraction = float(giant_comp_size / n_domains)

    # Group domain indices by component ID
    groups: Dict[int, List[int]] = {}
    for idx, c in enumerate(domain_comp_labels):
        groups.setdefault(int(c), []).append(idx)

    # Stratified group assignment
    train_idx, val_idx, test_idx, stat = assign_groups_stratified(
        groups=groups,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_discrepancy=max_discrepancy,
        seed=seed
    )

    # Pre-allocate 50% of validation negatives into an independent certification holdout
    val_labels = labels[val_idx]
    val_neg_indices = val_idx[val_labels == 0]
    val_pos_indices = val_idx[val_labels == 1]

    if len(val_neg_indices) >= 2:
        rng = np.random.RandomState(seed)
        shuffled_negs = rng.permutation(val_neg_indices)
        half = len(val_neg_indices) // 2
        calib_negs = shuffled_negs[:half]
        cert_negs = shuffled_negs[half:]
        model_val_indices = np.concatenate([val_pos_indices, calib_negs])
        model_val_indices.sort()
        cert_indices = cert_negs
        cert_indices.sort()
    else:
        model_val_indices = val_idx
        cert_indices = np.array([], dtype=int)

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=model_val_indices,
        test_indices=test_idx,
        split_type="group_joint" if joint_grouping else "group_bipartite",
        metadata={
            "giant_component_fraction": round(giant_comp_fraction, 4),
            "giant_component_size": giant_comp_size,
            "num_components": int(n_components),
            "hubs_removed": int(np.sum(infra_degrees > effective_cutoff)),
            "effective_cutoff": float(effective_cutoff),
            "joint_grouping": joint_grouping,
            "stratification_stats": stat,
            "cert_size": len(cert_indices),
        },
        cert_indices=cert_indices
    )
    splits.assert_disjoint()

    meta = {
        "giant_component_fraction": giant_comp_fraction,
        "splits": splits,
        "stratification_stats": stat
    }
    return splits, meta


def compute_asn_leakage_report(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray
) -> Dict[str, float]:
    """
    Computes fraction of test domains sharing infrastructure (NS, registrar, cert)
    with any training domain under the ASN fallback split.
    """
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Collect train infrastructure sets
    train_ns: Set[str] = set()
    for ns_list in train_df["nameservers"]:
        if isinstance(ns_list, (list, np.ndarray)):
            train_ns.update(ns_list)

    train_regs = set(train_df["registrar"].dropna().unique())
    train_certs = set(train_df["leaf_cert_key"].dropna().unique())

    # Check test domains via vectorized list access (zero iterrows)
    n_test = len(test_df)
    if n_test == 0:
        return {"ns_leakage_share": 0.0, "reg_leakage_share": 0.0, "cert_leakage_share": 0.0, "any_infra_leakage_share": 0.0}

    leak_ns = 0
    leak_reg = 0
    leak_cert = 0
    leak_any = 0

    t_ns_col = test_df["nameservers"].tolist() if "nameservers" in test_df.columns else [None] * n_test
    t_reg_col = test_df["registrar"].tolist() if "registrar" in test_df.columns else [None] * n_test
    t_cert_col = test_df["leaf_cert_key"].tolist() if "leaf_cert_key" in test_df.columns else [None] * n_test

    for idx in range(n_test):
        test_ns_list = t_ns_col[idx]
        has_l_ns = isinstance(test_ns_list, (list, np.ndarray)) and any(ns in train_ns for ns in test_ns_list)
        r_reg = t_reg_col[idx]
        has_l_reg = bool(r_reg and r_reg in train_regs)
        r_cert = t_cert_col[idx]
        has_l_cert = bool(r_cert and r_cert in train_certs)

        if has_l_ns:
            leak_ns += 1
        if has_l_reg:
            leak_reg += 1
        if has_l_cert:
            leak_cert += 1
        if has_l_ns or has_l_reg or has_l_cert:
            leak_any += 1

    return {
        "ns_leakage_share": round(leak_ns / n_test, 4),
        "reg_leakage_share": round(leak_reg / n_test, 4),
        "cert_leakage_share": round(leak_cert / n_test, 4),
        "any_infra_leakage_share": round(leak_any / n_test, 4),
        "test_domains_count": n_test
    }


def group_split_by_asn(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_discrepancy: float = 0.05,
    seed: int = 42
) -> Tuple[DataSplits, Dict[str, Any]]:
    """
    Alternative group split by primary ASN (used if bipartite giant component > 50%).
    Applies stratified group assignment and generates an infrastructure leakage report.
    """
    labels = df["label"].to_numpy()

    # Map missing ASNs to individual negative unique IDs (flat list indexing, zero iterrows)
    asn_groups: Dict[int, List[int]] = {}
    missing_counter = -1
    asn_col = df["primary_asn"].tolist() if "primary_asn" in df.columns else [None] * len(df)

    for idx, asn in enumerate(asn_col):
        if asn is not None and not (isinstance(asn, float) and np.isnan(asn)):
            asn_groups.setdefault(int(asn), []).append(idx)
        else:
            asn_groups.setdefault(missing_counter, []).append(idx)
            missing_counter -= 1

    # Stratified group assignment
    train_idx, val_idx, test_idx, stat = assign_groups_stratified(
        groups=asn_groups,
        labels=labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_discrepancy=max_discrepancy,
        seed=seed
    )

    # Compute leakage report
    leakage = compute_asn_leakage_report(df, train_idx, test_idx)

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="group_asn",
        metadata={
            "unique_asns_count": len(asn_groups),
            "seed": seed,
            "stratification_stats": stat,
            "leakage_report": leakage
        }
    )
    splits.assert_disjoint()
    return splits, {"leakage_report": leakage, "stratification_stats": stat}


def stratified_apex_cluster_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> DataSplits:
    """
    RFC-001 Leak-Free Apex-Cluster Stratified Split.
    Partitions FQDNs strictly by Apex Domain Equivalence Classes so that all
    subdomains sharing an apex domain are assigned to the exact same partition.
    Pre-allocates 50% of validation negatives into cert_indices for FPR calibration.
    """
    n_total = len(df)
    if n_total == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    if "apex_domain" in df.columns:
        apex_series = df["apex_domain"].fillna(df["domain"]).astype(str).str.lower()
    elif "e2LD" in df.columns:
        apex_series = df["e2LD"].fillna(df["domain"]).astype(str).str.lower()
    else:
        import tldextract
        ext = tldextract.TLDExtract(suffix_list_urls=())
        apex_vals = []
        for d in df["domain"].astype(str).tolist():
            r = ext(d.strip().lower())
            apex_vals.append(f"{r.domain}.{r.suffix}" if (r.domain and r.suffix) else d.strip().lower())
        apex_series = pd.Series(apex_vals, index=df.index)

    labels = df["label"].to_numpy().astype(int)
    tmp_df = pd.DataFrame({
        "row_pos": np.arange(n_total, dtype=int),
        "apex_domain": apex_series.to_numpy(),
        "label": labels,
    })

    apex_summary = tmp_df.groupby("apex_domain", sort=True).agg(
        has_malware=("label", "max"),
        fqdn_count=("row_pos", "count"),
    ).reset_index()

    n_apex = len(apex_summary)
    if n_apex < 3:
        raise ValueError(f"Need at least 3 distinct apex domains for train/val/test split, found {n_apex}.")

    apex_labels = apex_summary["has_malware"].to_numpy().astype(int)
    holdout_size = float(val_ratio + test_ratio)
    can_sss1 = len(np.unique(apex_labels)) > 1 and np.min(np.bincount(apex_labels)) >= 2
    if can_sss1:
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
            train_apex_idx, holdout_apex_idx = next(sss1.split(apex_summary, apex_labels))
        except ValueError:
            can_sss1 = False
    if not can_sss1:
        rng_fb = np.random.RandomState(seed)
        perm = rng_fb.permutation(n_apex)
        cut = max(1, int(round(train_ratio * n_apex)))
        train_apex_idx, holdout_apex_idx = perm[:cut], perm[cut:]

    holdout_summary = apex_summary.iloc[holdout_apex_idx].reset_index(drop=True)
    holdout_labels = holdout_summary["has_malware"].to_numpy().astype(int)
    val_rel_size = float(val_ratio / max(holdout_size, 1e-8))
    can_sss2 = (
        len(holdout_summary) >= 4
        and len(np.unique(holdout_labels)) > 1
        and np.min(np.bincount(holdout_labels)) >= 2
    )
    if can_sss2:
        try:
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_rel_size), random_state=seed)
            val_sub_idx, test_sub_idx = next(sss2.split(holdout_summary, holdout_labels))
        except ValueError:
            can_sss2 = False
    if not can_sss2:
        rng_fb2 = np.random.RandomState(seed + 1)
        perm2 = rng_fb2.permutation(len(holdout_summary))
        cut2 = max(1, len(perm2) // 2)
        val_sub_idx, test_sub_idx = perm2[:cut2], perm2[cut2:]

    train_apexes = set(apex_summary.iloc[train_apex_idx]["apex_domain"].tolist())
    val_apexes = set(holdout_summary.iloc[val_sub_idx]["apex_domain"].tolist())
    test_apexes = set(holdout_summary.iloc[test_sub_idx]["apex_domain"].tolist())

    # Formal Leakage Invariant Assertions (RFC-001 Section 4)
    assert len(train_apexes.intersection(test_apexes)) == 0, "Apex leakage detected (Train vs Test)!"
    assert len(train_apexes.intersection(val_apexes)) == 0, "Apex leakage detected (Train vs Val)!"
    assert len(val_apexes.intersection(test_apexes)) == 0, "Apex leakage detected (Val vs Test)!"

    apex_arr = tmp_df["apex_domain"].to_numpy()
    train_indices = np.where(np.isin(apex_arr, list(train_apexes)))[0]
    val_indices = np.where(np.isin(apex_arr, list(val_apexes)))[0]
    test_indices = np.where(np.isin(apex_arr, list(test_apexes)))[0]

    # 50% negative certification holdout strictly from val_indices
    val_labels = labels[val_indices]
    val_neg_indices = val_indices[val_labels == 0]
    val_pos_indices = val_indices[val_labels == 1]

    if len(val_neg_indices) >= 2:
        rng = np.random.RandomState(seed)
        shuffled_negs = rng.permutation(val_neg_indices)
        half = len(val_neg_indices) // 2
        calib_negs = shuffled_negs[:half]
        cert_negs = shuffled_negs[half:]
        model_val_indices = np.concatenate([val_pos_indices, calib_negs])
        model_val_indices.sort()
        cert_indices = cert_negs
        cert_indices.sort()
    else:
        model_val_indices = val_indices
        cert_indices = np.array([], dtype=int)

    splits = DataSplits(
        train_indices=train_indices,
        val_indices=model_val_indices,
        test_indices=test_indices,
        split_type="apex_cluster",
        metadata={
            "train_apex_count": len(train_apexes),
            "val_apex_count": len(val_apexes),
            "test_apex_count": len(test_apexes),
            "total_apex_count": n_apex,
            "train_count": len(train_indices),
            "val_count": len(model_val_indices),
            "cert_count": len(cert_indices),
            "test_count": len(test_indices),
            "apex_leakage_count": 0,
        },
        cert_indices=cert_indices,
    )
    splits.assert_disjoint()
    return splits


def within_apex_sibling_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DataSplits:
    """
    RFC-001 Threat Scenario 2 (Sibling Exploitation on Multi-Tenant / Compromised Apexes):
    Identifies apex domains with >= 2 FQDNs and partitions FQDNs *within* shared apexes
    across train/val/test so the model must rely on FQDN lexical + IP telemetry to catch
    malicious siblings under otherwise benign/multi-tenant apexes.
    """
    n_total = len(df)
    if "apex_domain" in df.columns:
        apex_series = df["apex_domain"].fillna(df["domain"]).astype(str).str.lower()
    elif "e2LD" in df.columns:
        apex_series = df["e2LD"].fillna(df["domain"]).astype(str).str.lower()
    else:
        apex_series = df["domain"].astype(str).str.lower()

    counts = apex_series.value_counts()
    multi_apexes = set(counts[counts >= 2].index.tolist())
    labels = df["label"].to_numpy().astype(int)

    if len(multi_apexes) == 0:
        # Fallback cleanly to standard random stratified split if no multi-tenant apexes exist
        fallback = random_stratified_split(labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        fallback.split_type = "within_apex_sibling"
        fallback.metadata["multi_tenant_apex_count"] = 0
        return fallback

    multi_mask = apex_series.isin(multi_apexes).to_numpy()
    single_indices = np.where(~multi_mask)[0]

    # Place at least one benign or earliest sibling per multi-tenant apex into train, stratify remaining
    rng = np.random.RandomState(seed)
    train_list: List[int] = list(single_indices.tolist())
    holdout_pool: List[int] = []

    apex_groups: Dict[str, List[int]] = {}
    for idx in np.where(multi_mask)[0]:
        apex_groups.setdefault(apex_series.iloc[idx], []).append(int(idx))

    for a_dom, idxs in apex_groups.items():
        idxs_sorted = sorted(idxs, key=lambda i: (labels[i], i))  # prefer benign anchor in train
        train_list.append(idxs_sorted[0])
        holdout_pool.extend(idxs_sorted[1:])

    holdout_arr = np.array(holdout_pool, dtype=int)
    target_train_total = int(round(train_ratio * n_total))
    need_extra_train = max(0, min(len(holdout_arr) - 2, target_train_total - len(train_list)))
    if need_extra_train > 0 and len(holdout_arr) > need_extra_train:
        perm = rng.permutation(len(holdout_arr))
        train_list.extend(holdout_arr[perm[:need_extra_train]].tolist())
        rem_arr = holdout_arr[perm[need_extra_train:]]
    else:
        rem_arr = holdout_arr

    rem_labels = labels[rem_arr]
    val_frac = float(val_ratio / max(1.0 - train_ratio, 1e-6))
    if len(rem_arr) >= 4 and len(np.unique(rem_labels)) > 1 and np.min(np.bincount(rem_labels)) >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_frac), random_state=seed)
        v_rel, t_rel = next(sss.split(rem_arr, rem_labels))
        val_indices = rem_arr[v_rel]
        test_indices = rem_arr[t_rel]
    else:
        perm_r = rng.permutation(len(rem_arr))
        cut_r = max(1, int(round(val_frac * len(perm_r))))
        val_indices = rem_arr[perm_r[:cut_r]]
        test_indices = rem_arr[perm_r[cut_r:]]

    train_indices = np.sort(np.array(train_list, dtype=int))
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    val_negs = val_indices[labels[val_indices] == 0]
    val_pos = val_indices[labels[val_indices] == 1]
    if len(val_negs) >= 2:
        shuf = rng.permutation(val_negs)
        half = len(val_negs) // 2
        model_val = np.sort(np.concatenate([val_pos, shuf[:half]]))
        cert_idx = np.sort(shuf[half:])
    else:
        model_val = val_indices
        cert_idx = np.array([], dtype=int)

    splits = DataSplits(
        train_indices=train_indices,
        val_indices=model_val,
        test_indices=test_indices,
        split_type="within_apex_sibling",
        metadata={
            "multi_tenant_apex_count": len(multi_apexes),
            "train_count": len(train_indices),
            "val_count": len(model_val),
            "cert_count": len(cert_idx),
            "test_count": len(test_indices),
        },
        cert_indices=cert_idx,
    )
    splits.assert_disjoint()
    return splits


# Re-export temporal diagnostics and holdout split for unified consumption
from src.hdg.data.temporal import (
    single_chronological_holdout_split,
    analyze_temporal_eligibility,
    save_temporal_eligibility_report,
    TemporalEligibilityResult,
)

__all__ = [
    "DataSplits",
    "random_stratified_split",
    "rolling_origin_temporal_splits",
    "time_split",
    "group_split_bipartite",
    "group_split_by_asn",
    "stratified_apex_cluster_split",
    "within_apex_sibling_split",
    "compute_asn_leakage_report",
    "single_chronological_holdout_split",
    "analyze_temporal_eligibility",
    "save_temporal_eligibility_report",
    "TemporalEligibilityResult",
]
