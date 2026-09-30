"""
Leak-free dataset split generation for Malicious-HDG.
Implements:
1. Random stratified split (optimistic bound only).
2. Temporal split (train < val < test by t, conditional on time_split_valid).
3. Group split by bipartite connected components after hub pruning (degree > hub_degree).
4. Group split by primary ASN (fallback if giant component > 50%).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedShuffleSplit


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


def time_split(
    df: pd.DataFrame,
    time_split_valid: bool,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    min_positives: int = 5,
    min_negatives: int = 5
) -> DataSplits:
    """
    Strict temporal split: train < val < test by timestamp t.
    Only valid if time_split_valid is True and all windows have sufficient positive/negative samples.
    """
    if not time_split_valid:
        raise ValueError(
            "Temporal split requested but time_split_valid is False. "
            "Classes do not have overlapping longitudinal coverage; time experiments disabled."
        )

    # Sort strictly by timestamp
    df_sorted = df.sort_values(by=["t"]).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)

    # Validate window class balance
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        y_win = df_sorted.iloc[idx]["label"].to_numpy()
        pos = int(np.sum(y_win == 1))
        neg = int(np.sum(y_win == 0))
        if pos < min_positives or neg < min_negatives:
            raise ValueError(
                f"Time window '{name}' has insufficient class representation (pos={pos}, neg={neg}, min={min_positives}/{min_negatives}). "
                "Time window contains single-class bias."
            )

    t_train_max = df_sorted.iloc[train_idx[-1]]["t"]
    t_val_min = df_sorted.iloc[val_idx[0]]["t"]
    t_val_max = df_sorted.iloc[val_idx[-1]]["t"]
    t_test_min = df_sorted.iloc[test_idx[0]]["t"]

    splits = DataSplits(
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        split_type="time",
        metadata={
            "t_train_max": t_train_max,
            "t_val_min": t_val_min,
            "t_val_max": t_val_max,
            "t_test_min": t_test_min
        }
    )
    splits.assert_disjoint()
    return splits


def group_split_bipartite(
    df: pd.DataFrame,
    hub_degree: int = 500,
    hub_percentile: float = 99.9,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataSplits, Dict[str, Any]]:
    """
    Connected components of the domain-infrastructure bipartite graph AFTER removing hub nodes.
    Computes giant component fraction. If > 50%, reports warning.
    Guarantees no component/cluster spans multiple split sets.
    """
    n_domains = len(df)
    rng = np.random.default_rng(seed)

    # Map infrastructure entities to unique IDs
    infra_to_id: Dict[str, int] = {}
    edges_domain: List[int] = []
    edges_infra: List[int] = []

    for d_idx, row in df.iterrows():
        # Connect to IPs
        res_ips = row.get("resolved_ips")
        if res_ips is not None:
            for ip in res_ips:
                k = f"ip:{ip}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        # Connect to Nameservers
        ns_list = row.get("nameservers")
        if ns_list is not None:
            for ns in ns_list:
                k = f"ns:{ns}"
                if k not in infra_to_id:
                    infra_to_id[k] = len(infra_to_id)
                edges_domain.append(d_idx)
                edges_infra.append(infra_to_id[k])

        # Connect to Registrar
        reg = row.get("registrar")
        if reg:
            k = f"reg:{reg}"
            if k not in infra_to_id:
                infra_to_id[k] = len(infra_to_id)
            edges_domain.append(d_idx)
            edges_infra.append(infra_to_id[k])

        # Connect to Leaf Certificate
        cert = row.get("leaf_cert_key")
        if cert:
            k = f"cert:{cert}"
            if k not in infra_to_id:
                infra_to_id[k] = len(infra_to_id)
            edges_domain.append(d_idx)
            edges_infra.append(infra_to_id[k])

    n_infra = len(infra_to_id)
    if n_infra == 0 or len(edges_domain) == 0:
        # Fallback to random stratified if graph is completely disconnected
        return random_stratified_split(df["label"].to_numpy(), train_ratio, val_ratio, seed), {"giant_component_fraction": 0.0}

    # Count degrees of infra nodes
    infra_degrees = np.bincount(edges_infra, minlength=n_infra)
    perc_cutoff = np.percentile(infra_degrees, hub_percentile)
    effective_cutoff = min(hub_degree, perc_cutoff)

    # Filter out hub infra nodes
    valid_mask = np.array([infra_degrees[i] <= effective_cutoff for i in edges_infra])
    f_d = np.array(edges_domain)[valid_mask]
    f_i = np.array(edges_infra)[valid_mask]

    # Build bipartite adjacency: size (n_domains + n_infra)
    row_idx = np.concatenate([f_d, f_i + n_domains])
    col_idx = np.concatenate([f_i + n_domains, f_d])
    data_ones = np.ones(len(row_idx), dtype=int)
    adj = csr_matrix((data_ones, (row_idx, col_idx)), shape=(n_domains + n_infra, n_domains + n_infra))

    n_components, labels = connected_components(adj, directed=False)
    domain_comp_labels = labels[:n_domains]

    # Component size analysis
    comp_sizes = np.bincount(domain_comp_labels, minlength=n_components)
    giant_comp_size = int(np.max(comp_sizes))
    giant_comp_fraction = float(giant_comp_size / n_domains)

    # Group components into train / val / test sets
    unique_comps = np.unique(domain_comp_labels)
    rng.shuffle(unique_comps)

    train_target = int(n_domains * train_ratio)
    val_target = int(n_domains * val_ratio)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for comp in unique_comps:
        comp_members = np.where(domain_comp_labels == comp)[0]
        if len(train_indices) < train_target:
            train_indices.extend(comp_members.tolist())
        elif len(val_indices) < val_target:
            val_indices.extend(comp_members.tolist())
        else:
            test_indices.extend(comp_members.tolist())

    splits = DataSplits(
        train_indices=np.array(train_indices),
        val_indices=np.array(val_indices),
        test_indices=np.array(test_indices),
        split_type="group_bipartite",
        metadata={
            "giant_component_fraction": round(giant_comp_fraction, 4),
            "giant_component_size": giant_comp_size,
            "num_components": int(n_components),
            "hubs_removed": int(np.sum(infra_degrees > effective_cutoff)),
            "effective_cutoff": float(effective_cutoff)
        }
    )
    splits.assert_disjoint()

    metadata = {
        "giant_component_fraction": giant_comp_fraction,
        "splits": splits
    }
    return splits, metadata


def group_split_by_asn(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> DataSplits:
    """
    Alternative group split by primary ASN (used if bipartite giant component > 50%).
    Guarantees no ASN spans multiple split sets.
    """
    n_domains = len(df)
    rng = np.random.default_rng(seed)

    # Map missing ASNs to individual negative unique IDs
    asn_labels: List[int] = []
    missing_counter = -1
    for _, row in df.iterrows():
        asn = row.get("primary_asn")
        if asn is not None and not np.isnan(asn):
            asn_labels.append(int(asn))
        else:
            asn_labels.append(missing_counter)
            missing_counter -= 1

    asn_array = np.array(asn_labels)
    unique_asns = np.unique(asn_array)
    rng.shuffle(unique_asns)

    train_target = int(n_domains * train_ratio)
    val_target = int(n_domains * val_ratio)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for asn in unique_asns:
        members = np.where(asn_array == asn)[0]
        if len(train_indices) < train_target:
            train_indices.extend(members.tolist())
        elif len(val_indices) < val_target:
            val_indices.extend(members.tolist())
        else:
            test_indices.extend(members.tolist())

    splits = DataSplits(
        train_indices=np.array(train_indices),
        val_indices=np.array(val_indices),
        test_indices=np.array(test_indices),
        split_type="group_asn",
        metadata={
            "unique_asns_count": len(unique_asns),
            "seed": seed
        }
    )
    splits.assert_disjoint()
    return splits
