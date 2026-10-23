"""
Unit and regression tests for Dataset-Graph Row Alignment (Root Cause D).
Guarantees that dataframe rows, labels, split indices, and graph domain nodes match 1-to-1,
and that stale or misaligned cohorts are rejected with clear diagnostics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData

from src.hdg.data.graph import (
    save_graph_cohort_bundle,
    load_aligned_graph_cohort,
    build_base_graph
)


def test_save_and_load_aligned_graph_cohort(tmp_path: Path):
    """Verifies that saving and loading an aligned graph-cohort bundle preserves 100% integrity."""
    df = pd.DataFrame({
        "domain": [f"domain-{i}.com" for i in range(20)],
        "label": [0, 1] * 10,
        "resolved_ips": [[f"10.0.0.{i % 4}"] for i in range(20)],
        "nameservers": [["ns1.com"]] * 20,
        "registrar": ["Reg1"] * 20
    })

    graph = build_base_graph(df)

    # Save bundle
    manifest = save_graph_cohort_bundle(
        graph=graph,
        df=df,
        output_dir=tmp_path,
        config={"test": True}
    )

    assert manifest["domain_count"] == 20
    assert (tmp_path / "heterodata.pt").exists()
    assert (tmp_path / "domains_cohort.parquet").exists()
    assert (tmp_path / "graph_manifest.json").exists()
    assert (tmp_path / "id_maps.json").exists()

    # Load bundle
    loaded_graph, loaded_df, loaded_manifest = load_aligned_graph_cohort(tmp_path, require_manifest=True)

    assert loaded_graph["domain"].num_nodes == len(loaded_df)
    assert len(loaded_df) == 20
    assert list(loaded_df["domain"]) == list(df["domain"])
    assert torch.equal(loaded_graph["domain"].y, torch.tensor(df["label"].to_numpy(), dtype=torch.long))
    assert loaded_manifest["manifest_id"] == manifest["manifest_id"]


def test_rejection_of_misaligned_node_count(tmp_path: Path):
    """Root Cause D: Loading a graph with different node count than dataframe must fail loudly."""
    df = pd.DataFrame({
        "domain": [f"domain-{i}.com" for i in range(20)],
        "label": [0, 1] * 10
    })
    graph = build_base_graph(df)

    save_graph_cohort_bundle(graph=graph, df=df, output_dir=tmp_path)

    # Corrupt cohort dataframe by truncating to 15 rows
    truncated_df = df.iloc[:15].copy()
    truncated_df.to_parquet(tmp_path / "domains_cohort.parquet", index=False)

    with pytest.raises(ValueError, match="FATAL ALIGNMENT MISMATCH: graph\\['domain'\\]\\.num_nodes"):
        load_aligned_graph_cohort(tmp_path)


def test_rejection_of_corrupted_labels(tmp_path: Path):
    """Root Cause D: Loading a cohort with inverted or corrupted labels must fail loudly."""
    df = pd.DataFrame({
        "domain": [f"domain-{i}.com" for i in range(10)],
        "label": [0, 1] * 5
    })
    graph = build_base_graph(df)
    save_graph_cohort_bundle(graph=graph, df=df, output_dir=tmp_path)

    # Corrupt labels in parquet
    corrupted_df = df.copy()
    corrupted_df["label"] = 1  # All positive
    corrupted_df.to_parquet(tmp_path / "domains_cohort.parquet", index=False)

    with pytest.raises(ValueError, match="FATAL LABEL MISMATCH"):
        load_aligned_graph_cohort(tmp_path)


def test_deterministic_cohort_sampling():
    """Verifies that random sampling with random_state=42 produces identical ordered cohorts."""
    domains = [f"domain-{i:04d}.org" for i in range(500)]
    df = pd.DataFrame({"domain": domains, "val": range(500)})

    sample1 = df.sample(n=50, random_state=42).reset_index(drop=True)
    sample2 = df.sample(n=50, random_state=42).reset_index(drop=True)

    pd.testing.assert_frame_equal(sample1, sample2)
    assert list(sample1["domain"]) == list(sample2["domain"])
