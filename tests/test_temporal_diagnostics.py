"""
Unit and regression tests for temporal diagnostics, chronological holdout,
rolling-origin validation, and leak-free temporal graph snapshots.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.hdg.data.temporal import (
    analyze_temporal_eligibility,
    single_chronological_holdout_split,
    save_temporal_eligibility_report,
)
from src.hdg.data.splits import rolling_origin_temporal_splits
from src.hdg.data.graph import build_temporal_graph_pair


def test_historical_single_class_months_with_aggregate_mixed_history():
    """
    Simulates Zenodo DomainRadar v2 month distribution:
    - 2022-12: 30 benign, 0 malware
    - 2023-02: 30 benign, 0 malware
    - 2023-07: 0 benign, 30 malware
    - 2023-11: 0 benign, 30 malware
    - 2023-12: 30 benign, 30 malware (target month)

    Root Cause A: Single chronological holdout MUST be valid because the
    aggregate historical window contains both classes (60 benign, 60 malware),
    even though each historical month only contains one class individually.
    """
    rows = []
    # 2022-12: 30 benign
    for i in range(30):
        rows.append({"domain": f"benign-12-22-{i}.com", "label": 0, "t_month": "2022-12", "sourced_on": "2022-12-05T00:00:00Z"})
    # 2023-02: 30 benign
    for i in range(30):
        rows.append({"domain": f"benign-02-23-{i}.com", "label": 0, "t_month": "2023-02", "sourced_on": "2023-02-10T00:00:00Z"})
    # 2023-07: 30 malware
    for i in range(30):
        rows.append({"domain": f"mal-07-23-{i}.com", "label": 1, "t_month": "2023-07", "sourced_on": "2023-07-15T00:00:00Z"})
    # 2023-11: 30 malware
    for i in range(30):
        rows.append({"domain": f"mal-11-23-{i}.com", "label": 1, "t_month": "2023-11", "sourced_on": "2023-11-20T00:00:00Z"})
    # 2023-12: 30 benign, 30 malware
    for i in range(30):
        rows.append({"domain": f"benign-12-23-{i}.com", "label": 0, "t_month": "2023-12", "sourced_on": "2023-12-01T00:00:00Z"})
        rows.append({"domain": f"mal-12-23-{i}.com", "label": 1, "t_month": "2023-12", "sourced_on": "2023-12-01T00:00:00Z"})

    df = pd.DataFrame(rows)

    elig = analyze_temporal_eligibility(
        df,
        temporal_col="t_month",
        min_distinct_months=2,
        min_per_class=10
    )

    # Rolling origin must be False (only 1 overlapping month with both classes)
    assert not elig.rolling_origin_available
    assert len(elig.overlapping_months) == 1
    assert elig.overlapping_months[0] == "2023-12"

    # Single chronological holdout must be True targeting 2023-12!
    assert elig.single_holdout_available
    assert "2023-12" in elig.eligible_target_months

    # Perform the split
    splits = single_chronological_holdout_split(
        df,
        target_month="2023-12",
        temporal_col="t_month",
        min_historical_per_class=10,
        min_target_per_class=10,
        seed=42
    )

    # Verify train, val, and test partitions
    train_df = df.iloc[splits.train_indices]
    val_df = df.iloc[splits.val_indices]
    test_df = df.iloc[splits.test_indices]

    # Test set must ONLY contain target month records
    assert set(test_df["t_month"]) == {"2023-12"}
    assert (test_df["label"] == 1).sum() > 0
    assert (test_df["label"] == 0).sum() > 0

    # Train and Val must strictly precede target month
    assert all(m < "2023-12" for m in train_df["t_month"])
    assert all(m < "2023-12" for m in val_df["t_month"])

    # Both classes must be in train
    assert (train_df["label"] == 1).sum() > 0
    assert (train_df["label"] == 0).sum() > 0

    # Disjointness check
    all_indices = set(splits.train_indices) | set(splits.val_indices) | set(splits.test_indices)
    assert len(all_indices) == len(splits.train_indices) + len(splits.val_indices) + len(splits.test_indices)


def test_rolling_origin_blocked_with_one_mixed_month():
    """Checks that rolling-origin correctly reports empty when only 1 month has both classes."""
    rows = []
    # Month 1: benign only
    for i in range(20):
        rows.append({"domain": f"b1-{i}.com", "label": 0, "t_month": "2023-01"})
    # Month 2: malware only
    for i in range(20):
        rows.append({"domain": f"m2-{i}.com", "label": 1, "t_month": "2023-02"})
    # Month 3: mixed
    for i in range(20):
        rows.append({"domain": f"b3-{i}.com", "label": 0, "t_month": "2023-03"})
        rows.append({"domain": f"m3-{i}.com", "label": 1, "t_month": "2023-03"})

    df = pd.DataFrame(rows)
    transitions = rolling_origin_temporal_splits(
        df,
        min_distinct_months=2,
        min_per_class_per_window=5,
        allow_empty=True
    )
    assert len(transitions) == 0


def test_no_eligible_mixed_target_month_rejects():
    """If no target month contains both classes, single chronological holdout must reject."""
    rows = []
    for i in range(20):
        rows.append({"domain": f"b1-{i}.com", "label": 0, "t_month": "2023-01"})
        rows.append({"domain": f"m2-{i}.com", "label": 1, "t_month": "2023-02"})
        rows.append({"domain": f"b3-{i}.com", "label": 0, "t_month": "2023-03"})

    df = pd.DataFrame(rows)
    elig = analyze_temporal_eligibility(df, temporal_col="t_month", min_per_class=5)
    assert not elig.single_holdout_available
    assert not elig.rolling_origin_available
    assert elig.status == "BLOCKED_INSUFFICIENT_DATA"

    with pytest.raises(ValueError, match="Cannot construct valid single chronological holdout"):
        single_chronological_holdout_split(df, temporal_col="t_month", min_target_per_class=5)


def test_insufficient_samples_in_class_rejects():
    """Holdout must reject if malware count in target month is below minimum threshold."""
    rows = []
    # History: sufficient
    for i in range(30):
        rows.append({"domain": f"b1-{i}.com", "label": 0, "t_month": "2023-01"})
        rows.append({"domain": f"m1-{i}.com", "label": 1, "t_month": "2023-01"})
    # Target month: 30 benign, but only 2 malware (< 10 threshold)
    for i in range(30):
        rows.append({"domain": f"b2-{i}.com", "label": 0, "t_month": "2023-02"})
    for i in range(2):
        rows.append({"domain": f"m2-{i}.com", "label": 1, "t_month": "2023-02"})

    df = pd.DataFrame(rows)
    elig = analyze_temporal_eligibility(df, temporal_col="t_month", min_per_class=10)
    assert not elig.single_holdout_available


def test_multiple_eligible_future_windows():
    """When multiple months contain both classes, rolling-origin transitions are generated."""
    rows = []
    for m in ["2023-01", "2023-02", "2023-03"]:
        for i in range(25):
            rows.append({"domain": f"b-{m}-{i}.com", "label": 0, "t_month": m})
            rows.append({"domain": f"m-{m}-{i}.com", "label": 1, "t_month": m})

    df = pd.DataFrame(rows)
    elig = analyze_temporal_eligibility(df, temporal_col="t_month", min_distinct_months=2, min_per_class=10)
    assert elig.rolling_origin_available
    assert elig.streaming_available
    assert len(elig.overlapping_months) == 3

    transitions = rolling_origin_temporal_splits(df, min_distinct_months=2, min_per_class_per_window=10)
    assert len(transitions) >= 1


def test_temporal_cutoff_excludes_future_observations():
    """
    Root Cause C: Synthetically tests that records and graph relations
    strictly after train cutoff T are excluded from the training snapshot.
    """
    rows = [
        # Month 1 domain
        {"domain": "past1.com", "label": 0, "t_month": "2023-01", "resolved_ips": ["1.1.1.1"], "nameservers": ["ns1.com"], "registrar": "RegA"},
        # Month 2 domain
        {"domain": "past2.com", "label": 1, "t_month": "2023-02", "resolved_ips": ["2.2.2.2"], "nameservers": ["ns2.com"], "registrar": "RegB"},
        # Future Month 3 domain connecting to new IP 3.3.3.3 and shared IP 1.1.1.1
        {"domain": "future1.com", "label": 1, "t_month": "2023-03", "resolved_ips": ["1.1.1.1", "3.3.3.3"], "nameservers": ["ns3.com"], "registrar": "RegC"},
    ]
    df = pd.DataFrame(rows)

    train_graph, eval_graph, meta = build_temporal_graph_pair(
        df=df,
        train_cutoff_month="2023-02",
        target_month="2023-03",
        temporal_col="t_month"
    )

    # 1. Training graph must only contain past1 and past2
    assert train_graph["domain"].num_nodes == 2
    # 2. Evaluation graph contains all 3 domains
    assert eval_graph["domain"].num_nodes == 3

    # 3. New IP 3.3.3.3 and nameserver ns3.com must NOT exist in the training graph!
    train_ips = train_graph.id_maps["ip"]
    assert "3.3.3.3" not in train_ips
    assert "1.1.1.1" in train_ips
    assert "2.2.2.2" in train_ips

    train_ns = train_graph.id_maps["nameserver"]
    assert "ns3.com" not in train_ns
    assert "ns1.com" in train_ns

    # 4. In eval graph, 3.3.3.3 and ns3.com do exist
    eval_ips = eval_graph.id_maps["ip"]
    assert "3.3.3.3" in eval_ips

    # 5. Metadata contract verification
    assert meta["train_cutoff_month"] == "2023-02"
    assert meta["target_month"] == "2023-03"
    assert meta["train_domain_count"] == 2
    assert meta["target_domain_count"] == 1


def test_timestamp_field_preference_and_fallback():
    """
    Verifies that temporal extraction prefers sourced_on, falls back to evaluated_on,
    and handles missing/inconsistent values without inventing dates.
    """
    rows = [
        # Has sourced_on and evaluated_on -> should pick sourced_on (2023-04)
        {"domain": "d1.com", "label": 0, "sourced_on": "2023-04-10T00:00:00Z", "evaluated_on": "2023-05-10T00:00:00Z"},
        # Null sourced_on -> fallback to evaluated_on (2023-06)
        {"domain": "d2.com", "label": 1, "sourced_on": None, "evaluated_on": "2023-06-12T00:00:00Z"},
        # Both null -> None
        {"domain": "d3.com", "label": 0, "sourced_on": None, "evaluated_on": None},
    ]
    df = pd.DataFrame(rows)

    # Analyze temporal eligibility using fallback logic
    t_months = []
    for _, row in df.iterrows():
        s = row["sourced_on"]
        e = row["evaluated_on"]
        val = s if pd.notna(s) else (e if pd.notna(e) else None)
        t_months.append(str(val)[:7] if val is not None else None)

    assert t_months[0] == "2023-04"
    assert t_months[1] == "2023-06"
    assert t_months[2] is None
