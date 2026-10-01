"""
Unit tests for relational GNNGuard mechanism in Malicious-HDG.
Verifies:
1. Edge pruning drops edges with endpoint cosine similarity below prune_threshold.
2. Attention/edge weights sum to 1.0 per destination node.
3. Gradient flows back to the learnable beta layer-memory parameter.
"""

import pytest
import torch
from torch_geometric.utils import scatter

from src.hdg.models.hetero_gnn import GNNGuardLayer, HeteroGNN


def test_gnnguard_edge_pruning() -> None:
    """Verifies that dissimilar edges are assigned zero weight."""
    edge_types = [("domain", "resolves_to", "ip")]
    layer = GNNGuardLayer(edge_types=edge_types, hidden_dim=4, prune_threshold=0.5)

    # Node embeddings:
    # Edge 0: u=0, v=0 -> identical vectors (cosine sim = 1.0)
    # Edge 1: u=1, v=0 -> orthogonal/opposite vectors (cosine sim <= 0.0)
    h_domain = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float)

    h_ip = torch.tensor([
        [1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0, 1],  # source domains
        [0, 0]   # target ip
    ], dtype=torch.long)

    h_dict = {"domain": h_domain, "ip": h_ip}
    edge_dict = {("domain", "resolves_to", "ip"): edge_index}

    h_out, weights_dict = layer(h_dict, edge_dict)
    key = "domain__resolves_to__ip"
    weights = weights_dict[key]

    # Edge 0 should survive (sim=1.0 >= 0.5), Edge 1 should be pruned (sim=0.0 < 0.5)
    assert weights[0] > 0.99
    assert weights[1] == 0.0


def test_gnnguard_weights_sum_to_one() -> None:
    """Verifies row-normalization: weights of incoming edges to each destination sum to 1.0."""
    edge_types = [("domain", "resolves_to", "ip")]
    layer = GNNGuardLayer(edge_types=edge_types, hidden_dim=4, prune_threshold=0.0)

    # 4 domains connecting to 2 IPs with strictly positive vectors
    h_domain = torch.rand(4, 4) + 0.5
    h_ip = torch.rand(2, 4) + 0.5

    # IP 0 has 3 incoming edges (domains 0, 1, 2)
    # IP 1 has 1 incoming edge (domain 3)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 0, 1]
    ], dtype=torch.long)

    h_dict = {"domain": h_domain, "ip": h_ip}
    edge_dict = {("domain", "resolves_to", "ip"): edge_index}

    h_out, weights_dict = layer(h_dict, edge_dict)
    key = "domain__resolves_to__ip"
    weights = weights_dict[key]

    sums = scatter(weights, edge_index[1], dim=0, dim_size=2, reduce="sum")
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)


def test_gnnguard_gradient_flows_to_beta() -> None:
    """Verifies that gradients propagate to learnable beta parameter through layer-wise memory."""
    torch.manual_seed(42)
    edge_types = [("domain", "resolves_to", "ip")]
    layer = GNNGuardLayer(edge_types=edge_types, hidden_dim=4, prune_threshold=0.0)

    h_domain = torch.randn(2, 4, requires_grad=True)
    h_ip = torch.randn(1, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)

    h_dict = {"domain": h_domain, "ip": h_ip}
    edge_dict = {("domain", "resolves_to", "ip"): edge_index}
    prev_weights = {"domain__resolves_to__ip": torch.tensor([0.2, 0.8])}

    h_out, weights_dict = layer(h_dict, edge_dict, prev_weights_dict=prev_weights)
    target = torch.tensor([1.0, -1.0, 2.0, -0.5])
    loss = (h_out["ip"][0] * target).sum()
    loss.backward()

    key = "domain__resolves_to__ip"
    beta_param = layer.betas[key]
    assert beta_param.grad is not None
    assert torch.abs(beta_param.grad).item() > 0.0
