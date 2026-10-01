"""
Unit tests for 2-hop ego-subgraph extraction and CPU inference equivalence in Malicious-HDG.
Verifies that evaluating HeteroGNN on an extracted 2-hop ego-subgraph yields predictions
strictly identical (within 1e-5) to slicing predictions from a full-graph forward pass.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.latency import build_ego_subgraph


def test_ego_subgraph_forward_pass_equivalence(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies ego-subgraph forward pass mathematically equals full-graph forward pass."""
    data = minimal_synthetic_heterodata
    torch.manual_seed(42)

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        num_layers=2
    )
    model.eval()

    # 1. Full-graph forward pass
    with torch.no_grad():
        full_out = model(data.x_dict, data.edge_index_dict)
        full_probs = F.softmax(full_out, dim=-1)[:, 1].cpu().numpy()

    # 2. Ego-subgraph forward passes for individual domains
    for target_domain_idx in range(data["domain"].num_nodes):
        ego_data, local_targets = build_ego_subgraph(
            data=data,
            target_domain_indices=[target_domain_idx],
            num_hops=2
        )
        assert len(local_targets) == 1

        with torch.no_grad():
            sub_out = model(ego_data.x_dict, ego_data.edge_index_dict)
            sub_probs = F.softmax(sub_out, dim=-1)[:, 1].cpu().numpy()

        local_idx = local_targets[0]
        ego_pred = sub_probs[local_idx]
        full_pred = full_probs[target_domain_idx]

        # Must match within machine floating-point tolerance
        assert np.isclose(ego_pred, full_pred, atol=1e-5), (
            f"Equivalence failed for domain {target_domain_idx}: ego={ego_pred:.6f}, full={full_pred:.6f}"
        )


def test_ego_subgraph_batch_equivalence(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies mini-batch ego-subgraph extraction matches individual full-graph outputs."""
    data = minimal_synthetic_heterodata
    torch.manual_seed(42)

    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}
    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        num_layers=2
    )
    model.eval()

    with torch.no_grad():
        full_out = model(data.x_dict, data.edge_index_dict)
        full_probs = F.softmax(full_out, dim=-1)[:, 1].cpu().numpy()

    batch_targets = [1, 4, 7]
    ego_data, local_targets = build_ego_subgraph(data, batch_targets, num_hops=2)

    with torch.no_grad():
        sub_out = model(ego_data.x_dict, ego_data.edge_index_dict)
        sub_probs = F.softmax(sub_out, dim=-1)[:, 1].cpu().numpy()

    for orig_idx, loc_idx in zip(batch_targets, local_targets):
        ego_pred = sub_probs[loc_idx]
        full_pred = full_probs[orig_idx]
        assert np.isclose(ego_pred, full_pred, atol=1e-5)
