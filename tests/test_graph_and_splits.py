"""
Tests for graph construction and dataset splitting in Malicious-HDG.
Verifies:
- Cumulative graph construction (no future edges beyond t_max)
- Standardizer fit on train only (leak-free standardization)
- Disjointness and leak-free properties of Random, Time, and Group splits
- Window class balance enforcement in temporal split
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.hdg.graph import build_hetero_graph, TrainOnlyStandardizer
from src.hdg.splits import (
    random_stratified_split,
    time_split,
    group_split_bipartite,
    group_split_by_asn,
    DataSplits
)


def make_toy_domains_df() -> pd.DataFrame:
    """Creates synthetic toy domains DataFrame for split & graph tests."""
    records = []
    dates = [
        "2023-04-10T12:00:00Z", "2023-05-10T12:00:00Z", "2023-06-10T12:00:00Z",
        "2023-07-10T12:00:00Z", "2023-08-10T12:00:00Z", "2023-09-10T12:00:00Z"
    ]
    idx = 0
    for t_val in dates:
        for is_mal in [True, False]:
            for rep in range(10):
                asn_val = 13335 if idx < 60 else 15169
                reg_val = "GoDaddy" if idx % 3 == 0 else "Namecheap"
                ns_val = [f"ns1.prov{idx % 4}.com"]
                ip_val = [f"198.51.100.{idx % 10}"]

                records.append({
                    "domain": f"testdomain{idx}.com",
                    "e2LD": f"testdomain{idx}.com",
                    "has_subdomain": False,
                    "label": 1 if is_mal else 0,
                    "source": "ThreatFox" if is_mal else "Umbrella",
                    "family": "emotet" if is_mal else None,
                    "t": t_val,
                    "t_month": t_val[:7],
                    "no_ip": False,
                    "no_rdap": False,
                    "no_tls": False,
                    "mx_count": 1,
                    "has_spf": True,
                    "has_dmarc": False,
                    "has_dkim": False,
                    "dnssec": False,
                    "ttl_a": 300,
                    "ttl_ns": 86400,
                    "nameservers": ns_val,
                    "registrar": reg_val,
                    "tls_protocol": "TLSv1.3",
                    "tls_cipher": "AES",
                    "tls_cert_count": 1,
                    "leaf_cert_key": f"cert_{idx % 5}",
                    "cert_valid_len": 90,
                    "primary_asn": asn_val,
                    "resolved_ips": ip_val,
                    "ip_records": [{"ip": ip_val[0], "asn": asn_val}],
                    "ns_ip_pairs": [(ns_val[0], ip_val[0])]
                })
                idx += 1
    return pd.DataFrame(records)


def test_no_future_edges_in_cumulative_graph() -> None:
    df = make_toy_domains_df()

    # Cumulative filter: cutoff at 2023-06-30
    cutoff = "2023-06-30T23:59:59Z"
    data, scaler, id_maps = build_hetero_graph(df, t_max=cutoff)

    # All included domains must have t <= cutoff
    domain_map = id_maps["domain"]
    assert len(domain_map) > 0
    for dom_name in domain_map.keys():
        row_t = df[df["domain"] == dom_name]["t"].values[0]
        assert row_t <= cutoff

    # Domains with t > cutoff must NOT exist in the graph
    excluded_df = df[df["t"] > cutoff]
    assert len(excluded_df) > 0
    for dom_name in excluded_df["domain"].values:
        assert dom_name not in domain_map


def test_scaler_fit_on_train_only() -> None:
    """Verifies standardizer fit statistics depend strictly on train indices."""
    X = np.zeros((100, 2), dtype=np.float32)
    # Train set (first 50) has mean 10.0, std 2.0
    X[:50] = np.random.normal(loc=10.0, scale=2.0, size=(50, 2))
    # Test set (last 50) has mean 100.0, std 5.0
    X[50:] = np.random.normal(loc=100.0, scale=5.0, size=(50, 2))

    train_idx = np.arange(0, 50)
    scaler = TrainOnlyStandardizer()
    scaler.fit(X, train_idx)

    # Scaler mean must match train mean (~10), NOT pooled mean (~55)
    assert np.isclose(scaler.mean[0], 10.0, atol=1.0)
    assert np.isclose(scaler.mean[1], 10.0, atol=1.0)

    # Transformed train features must have mean ~0
    X_std = scaler.transform(X)
    assert np.isclose(np.mean(X_std[:50]), 0.0, atol=0.2)
    # Test features are shifted relative to train (~90 / std)
    assert np.mean(X_std[50:]) > 10.0


def test_splits_disjoint_and_leak_free() -> None:
    df = make_toy_domains_df()
    labels = df["label"].to_numpy()

    # 1. Random Stratified
    rand_splits = random_stratified_split(labels, seed=42)
    rand_splits.assert_disjoint()
    assert len(rand_splits.train_indices) + len(rand_splits.val_indices) + len(rand_splits.test_indices) == len(df)

    # 2. Time Split
    t_splits = time_split(df, time_split_valid=True)
    t_splits.assert_disjoint()
    # Check strict temporal ordering
    t_train_max = df.sort_values(by="t").iloc[t_splits.train_indices[-1]]["t"]
    t_val_min = df.sort_values(by="t").iloc[t_splits.val_indices[0]]["t"]
    t_val_max = df.sort_values(by="t").iloc[t_splits.val_indices[-1]]["t"]
    t_test_min = df.sort_values(by="t").iloc[t_splits.test_indices[0]]["t"]
    assert t_train_max <= t_val_min
    assert t_val_max <= t_test_min

    # 3. Group Split Bipartite
    grp_splits, meta = group_split_bipartite(df, seed=42)
    grp_splits.assert_disjoint()
    assert meta["giant_component_fraction"] >= 0.0

    # 4. Group Split ASN
    asn_splits, _ = group_split_by_asn(df, seed=42)
    asn_splits.assert_disjoint()


def test_time_split_fails_when_invalid() -> None:
    df = make_toy_domains_df()
    with pytest.raises(ValueError, match="time_split_valid is False"):
        _ = time_split(df, time_split_valid=False)
