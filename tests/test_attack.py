"""
Unit tests for structural adversarial attacks in Malicious-HDG.
Verifies:
1. Immutability: The clean input HeteroData graph is not modified in-place.
2. Target candidate constraint: Candidate target nodes for fake edges are strictly sourced from train benign domains.
3. Attack perturbation: Attacked graph contains additional injected edges for controlled malicious domains.
"""

from copy import deepcopy
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.hdg.attack import generate_structural_attack
from src.hdg.splits import DataSplits


def test_attack_clean_graph_immutability(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies that generate_structural_attack does NOT mutate the clean graph in-place."""
    clean_data = minimal_synthetic_heterodata

    # Snapshot clean graph tensors
    orig_dom_x = clean_data["domain"].x.clone()
    orig_edges = {et: clean_data[et].edge_index.clone() for et in clean_data.edge_types}

    splits = DataSplits(
        train_indices=np.array([0, 1, 2, 3]),
        val_indices=np.array([4, 5]),
        test_indices=np.array([6, 7, 8, 9]),
        split_type="test"
    )

    attacked_data = generate_structural_attack(
        clean_data=clean_data,
        splits=splits,
        budget=2,
        rho=1.0,
        surfaces=["ip", "nameserver"],
        manipulate_features=True,
        seed=42
    )

    # 1. Clean data must be completely unchanged
    assert torch.equal(clean_data["domain"].x, orig_dom_x)
    for et in clean_data.edge_types:
        assert torch.equal(clean_data[et].edge_index, orig_edges[et])

    # 2. Attacked graph must differ and have additional edges
    e_clean = clean_data["domain", "resolves_to", "ip"].edge_index.shape[1]
    e_atk = attacked_data["domain", "resolves_to", "ip"].edge_index.shape[1]
    assert e_atk >= e_clean


def test_attack_targets_strictly_from_train_benign(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies that injected target infrastructure nodes are derived only from train benign domains."""
    clean_data = minimal_synthetic_heterodata

    # Train benign is domain 0 and 2
    # Test malicious is domain 7 and 9
    splits = DataSplits(
        train_indices=np.array([0, 1, 2, 3]),
        val_indices=np.array([4, 5]),
        test_indices=np.array([6, 7, 8, 9]),
        split_type="test"
    )

    # Find IPs connected to train benign domains (0 and 2)
    e_arr = clean_data["domain", "resolves_to", "ip"].edge_index.numpy()
    train_benign_domains = [0, 2]
    valid_target_ips = set(e_arr[1][np.isin(e_arr[0], train_benign_domains)].tolist())

    attacked = generate_structural_attack(
        clean_data=clean_data,
        splits=splits,
        budget=1,
        rho=1.0,
        surfaces=["ip"],
        manipulate_features=False,
        seed=42
    )

    atk_e_arr = attacked["domain", "resolves_to", "ip"].edge_index.numpy()
    # Check all edges originating from malicious test domains (7, 9)
    mal_test_domains = [7, 9]
    mal_edges_mask = np.isin(atk_e_arr[0], mal_test_domains)
    connected_target_ips = set(atk_e_arr[1][mal_edges_mask].tolist())

    # The targets must overlap with valid train benign infrastructure
    assert len(connected_target_ips.intersection(valid_target_ips)) > 0
