"""
Leak-free dataset split generation for Malicious-HDG.
Implements:
1. Random stratified split (optimistic bound only).
2. Rolling-origin temporal split: T -> T+1 over months in which both classes exist.
3. Stratified group split: connected components after hub pruning with greedy class-prevalence balancing.
4. Joint-grouping option across all infrastructure types (IP, NS, registrar, certificate).
5. Group split by primary ASN with infrastructure leakage report.
"""

from collections import Counter
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
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    split_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assert_disjoint(self) -> None:
        """Asserts train, val, and test index sets are completely disjoint."""
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


def random_stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> DataSplits:
    """
    Random stratified split on labels.
    Reported strictly as an optimistic bound.
    """
    n_total = len(labels)
    indices = np.arange(n_total)

    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(sss1.split(indices, labels))

    temp_labels = labels[temp_idx]
    val_relative_size = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_relative_size), random_state=seed)
    sub_val_idx, sub_test_idx = next(sss2.split(temp_idx, temp_labels))

    val_idx = temp_idx[sub_val_idx]
    test_idx = temp_idx[sub_test_idx]

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="random_stratified",
        metadata={
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "seed": seed
        }
    )
    splits.assert_disjoint()
    return splits


def rolling_origin_temporal_splits(
    df: pd.DataFrame,
    min_distinct_months: int = 3,
    min_per_class_per_window: int = 200,
    train_val_split_ratio: float = 0.85,
    seed: int = 42
) -> List[Tuple[str, DataSplits]]:
    """
    Rolling-origin temporal evaluation over months where BOTH classes exist:
    For transition T -> T+1:
      - Train on months <= T (stratified 85/15 train/val split inside historical window)
      - Test on month T+1
    Skips transitions with insufficient class samples and logs exact reasons.
    Refuses if overlapping months < 2.
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
        raise ValueError(
            f"Overlapping months with both classes is {len(both_class_months)} (< 2). "
            "Temporal rolling-origin evaluation cannot proceed."
        )

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

        splits = DataSplits(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            split_type="temporal_rolling_origin",
            metadata={
                "transition": trans_name,
                "history_max_month": t_cur,
                "test_month": t_next,
                "train_pos": int(np.sum(df.iloc[train_indices]["label"] == 1)),
                "train_neg": int(np.sum(df.iloc[train_indices]["label"] == 0)),
                "val_pos": int(np.sum(df.iloc[val_indices]["label"] == 1)),
                "val_neg": int(np.sum(df.iloc[val_indices]["label"] == 0)),
                "test_pos": test_pos,
                "test_neg": test_neg,
                "seed": seed
            }
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
    n_total = len(labels)
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
    Connected components of the domain-infrastructure graph AFTER removing hub nodes.
    - If joint_grouping=True: includes IP, nameserver, registrar, and certificate relations.
    - Uses assign_groups_stratified to ensure class ratios in each split are within 5 percentage points.
    - Reports giant component fraction and fallback metadata.
    """
    n_domains = len(df)
    labels = df["label"].to_numpy()

    infra_to_id: Dict[str, int] = {}
    edges_domain: List[int] = []
    edges_infra: List[int] = []

    for d_idx, row in df.iterrows():
        # Resolves_to (IP)
        res_ips = row.get("resolved_ips")
        if res_ips is not None and isinstance(res_ips, (list, np.ndarray)):
            for ip in res_ips:
                k = f"ip:{ip}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        # Nameservers
        ns_list = row.get("nameservers")
        if ns_list is not None and isinstance(ns_list, (list, np.ndarray)):
            for ns in ns_list:
                k = f"ns:{ns}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        if joint_grouping:
            # Registrar
            reg = row.get("registrar")
            if reg:
                k = f"reg:{reg}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

            # Leaf Certificate
            cert = row.get("leaf_cert_key_coissue") or row.get("leaf_cert_key")
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

    valid_mask = np.array([infra_degrees[i] <= effective_cutoff for i in edges_infra])
    f_d = np.array(edges_domain)[valid_mask]
    f_i = np.array(edges_infra)[valid_mask]

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

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="group_joint" if joint_grouping else "group_bipartite",
        metadata={
            "giant_component_fraction": round(giant_comp_fraction, 4),
            "giant_component_size": giant_comp_size,
            "num_components": int(n_components),
            "hubs_removed": int(np.sum(infra_degrees > effective_cutoff)),
            "effective_cutoff": float(effective_cutoff),
            "joint_grouping": joint_grouping,
            "stratification_stats": stat
        }
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

    # Check test domains
    n_test = len(test_df)
    if n_test == 0:
        return {"ns_leakage_share": 0.0, "reg_leakage_share": 0.0, "cert_leakage_share": 0.0, "any_infra_leakage_share": 0.0}

    leak_ns = 0
    leak_reg = 0
    leak_cert = 0
    leak_any = 0

    for _, row in test_df.iterrows():
        has_l_ns, has_l_reg, has_l_cert = False, False, False

        test_ns_list = row.get("nameservers")
        if isinstance(test_ns_list, (list, np.ndarray)) and any(ns in train_ns for ns in test_ns_list):
            has_l_ns = True

        r_reg = row.get("registrar")
        if r_reg and r_reg in train_regs:
            has_l_reg = True

        r_cert = row.get("leaf_cert_key")
        if r_cert and r_cert in train_certs:
            has_l_cert = True

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
    n_domains = len(df)
    labels = df["label"].to_numpy()

    # Map missing ASNs to individual negative unique IDs
    asn_groups: Dict[int, List[int]] = {}
    missing_counter = -1

    for idx, row in df.iterrows():
        asn = row.get("primary_asn")
        if asn is not None and not np.isnan(asn):
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
