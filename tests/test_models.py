"""
Unit tests for HeteroGNN architectures in Malicious-HDG.
Verifies:
1. Isolated Domain MLP mode (use_edges=False) executes without relying on edge indices.
2. Attn/HGTConv handles empty/missing edge types gracefully without crashing.
3. SAGE_Guard forward pass produces valid logits of shape [N_domains, 2].
"""

import pytest
import torch
from torch_geometric.data import HeteroData

from src.hdg.models.hetero_gnn import HeteroGNN


def test_mlp_no_edges_mode(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies that use_edges=False evaluates isolated domain features without touching graph edges."""
    data = minimal_synthetic_heterodata
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage",
        hidden_dim=32,
        out_dim=16,
        use_edges=False
    )
    model.eval()

    # Pass empty edge dict to ensure no graph convolution occurs
    empty_edge_dict = {et: torch.empty((2, 0), dtype=torch.long) for et in data.edge_types}

    with torch.no_grad():
        out = model(data.x_dict, empty_edge_dict)

    assert out.shape == (data["domain"].num_nodes, 2)
    assert not torch.isnan(out).any()


def test_hgt_empty_edges_guard(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies that HGTConv variant guards against None outputs and empty edge types."""
    data = minimal_synthetic_heterodata
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="attn",
        hidden_dim=32,
        out_dim=16,
        num_layers=2
    )
    model.eval()

    # Create edge dictionary with some empty edge types
    edge_dict = dict(data.edge_index_dict)
    if ("domain", "uses_cert", "certificate") in edge_dict:
        edge_dict[("domain", "uses_cert", "certificate")] = torch.empty((2, 0), dtype=torch.long)

    with torch.no_grad():
        out = model(data.x_dict, edge_dict)

    assert out.shape == (data["domain"].num_nodes, 2)
    assert not torch.isnan(out).any()


def test_sage_guard_forward(minimal_synthetic_heterodata: HeteroData) -> None:
    """Verifies that HeteroSAGEGuard computes valid domain logits."""
    data = minimal_synthetic_heterodata
    in_channels = {nt: data[nt].x.shape[1] for nt in data.node_types}

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        variant="sage_guard",
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        prune_threshold=0.1
    )
    model.eval()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)

    assert out.shape == (data["domain"].num_nodes, 2)
    assert not torch.isnan(out).any()
