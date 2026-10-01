"""
Unit tests for leak-free dataset split generation in Malicious-HDG.
Verifies:
1. Rolling-origin temporal splits maintain strict zero domain overlap between train and test.
2. Stratified group split balances class prevalence across partitions within 5 percentage points.
3. Disjointness assertion triggers correctly if an overlap is introduced.
"""

import numpy as np
import pandas as pd
import pytest

from src.hdg.splits import (
    DataSplits,
    rolling_origin_temporal_splits,
    assign_groups_stratified,
    group_split_bipartite,
    group_split_by_asn,
    compute_asn_leakage_report,
)


def test_rolling_origin_zero_overlap(minimal_domains_df: pd.DataFrame) -> None:
    """Verifies that rolling origin splits have completely disjoint train and test domain index sets."""
    df = minimal_domains_df
    transitions = rolling_origin_temporal_splits(
        df=df,
        min_distinct_months=2,
        min_per_class_per_window=1,
        seed=42
    )

    assert len(transitions) >= 1
    for trans_name, splits in transitions:
        splits.assert_disjoint()
        s_train = set(splits.train_indices.tolist())
        s_val = set(splits.val_indices.tolist())
        s_test = set(splits.test_indices.tolist())

        # No intersection between train/val history and test future
        assert len(s_train.intersection(s_test)) == 0
        assert len(s_val.intersection(s_test)) == 0

        # Check temporal order: test month is strictly greater than history max month
        meta = splits.metadata
        assert meta["test_month"] > meta["history_max_month"]


def test_group_split_stratified_balance() -> None:
    """Verifies that greedy group assignment maintains class prevalence within tolerance."""
    # 50 clusters of size 4 = 200 nodes total
    rng = np.random.default_rng(42)
    groups = {i: list(range(i * 4, (i + 1) * 4)) for i in range(50)}
    labels = np.zeros(200, dtype=int)
    for i in range(25):
        labels[i * 4 : (i + 1) * 4] = 1  # 25 malicious clusters, 25 benign clusters

    train_idx, val_idx, test_idx, stats = assign_groups_stratified(
        groups=groups,
        labels=labels,
        train_ratio=0.70,
        val_ratio=0.15,
        max_discrepancy=0.05,
        seed=42
    )

    overall_prev = np.mean(labels)
    train_prev = np.mean(labels[train_idx])
    val_prev = np.mean(labels[val_idx])
    test_prev = np.mean(labels[test_idx])

    assert abs(train_prev - overall_prev) <= 0.05
    assert abs(val_prev - overall_prev) <= 0.05
    assert abs(test_prev - overall_prev) <= 0.05


def test_splits_disjoint_assertion() -> None:
    """Verifies that assert_disjoint raises AssertionError when an overlap is forced."""
    leaky_splits = DataSplits(
        train_indices=np.array([0, 1, 2]),
        val_indices=np.array([3, 4]),
        test_indices=np.array([2, 5]),  # index 2 is in both train and test!
        split_type="test"
    )
    with pytest.raises(AssertionError, match="Leak detected"):
        leaky_splits.assert_disjoint()
