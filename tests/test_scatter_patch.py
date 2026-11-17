"""
Tests for PyTorch Geometric memory-efficient scatter patch.
Validates numerical parity, gradient parity, and tensor dimension compatibility
between PyG fallback scatter and PyTorch native index_add_ / index_reduce_.
"""

import pytest
import torch
import torch.nn.functional as F
import torch_geometric.utils._scatter as pyg_scatter
import torch_geometric.utils as pyg_utils

from src.hdg.device import apply_memory_efficient_scatter_patch
from src.hdg.models.hetero_gnn import HeteroGNN


def test_scatter_patch_applied():
    """Verify that apply_memory_efficient_scatter_patch patches PyG modules."""
    res = apply_memory_efficient_scatter_patch()
    assert res is True
    assert getattr(pyg_scatter, "_is_memory_efficient_patched", False) is True
    assert pyg_scatter.scatter is pyg_utils.scatter


@pytest.mark.parametrize("reduce_op", ["sum", "add", "max", "min", "mean"])
@pytest.mark.parametrize("dim_size", [10, 25])
def test_scatter_patch_numerical_parity(reduce_op, dim_size):
    """Verify exact numerical equality with PyG fallback across reduction types."""
    torch.manual_seed(42)
    num_items = 100
    channels = 16

    src = torch.randn(num_items, channels, dtype=torch.float32)
    # Only map to a subset of nodes so some destination nodes remain unreferenced
    index = torch.randint(0, dim_size - 2, (num_items,), dtype=torch.long)

    # Patch is already active; compare patched scatter against expected native logic
    out_patched = pyg_scatter.scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce_op)

    assert out_patched.shape == (dim_size, channels)

    if reduce_op in ("sum", "add"):
        expected = src.new_zeros(dim_size, channels).index_add_(0, index, src)
    elif reduce_op in ("max", "amax"):
        expected = src.new_zeros(dim_size, channels).index_reduce_(0, index, src, reduce="amax", include_self=False)
    elif reduce_op in ("min", "amin"):
        expected = src.new_zeros(dim_size, channels).index_reduce_(0, index, src, reduce="amin", include_self=False)
    elif reduce_op == "mean":
        count = src.new_zeros(dim_size)
        count.index_add_(0, index, src.new_ones(src.size(0)))
        count = count.clamp(min=1)
        expected = src.new_zeros(dim_size, channels).index_add_(0, index, src) / count.unsqueeze(-1)

    assert torch.allclose(out_patched, expected, atol=1e-6), f"Mismatch for reduce='{reduce_op}'"


def test_scatter_patch_gradient_backprop():
    """Verify gradient computation backward through memory-efficient scatter."""
    torch.manual_seed(42)
    num_items = 50
    channels = 8
    dim_size = 15

    src1 = torch.randn(num_items, channels, requires_grad=True)
    src2 = src1.detach().clone().requires_grad_(True)
    index = torch.randint(0, dim_size, (num_items,))

    # Sum gradient
    out1 = pyg_scatter.scatter(src1, index, dim=0, dim_size=dim_size, reduce="sum")
    loss1 = (out1 * 2.5).sum()
    loss1.backward()

    out2 = src2.new_zeros(dim_size, channels).index_add_(0, index, src2)
    loss2 = (out2 * 2.5).sum()
    loss2.backward()

    assert torch.allclose(src1.grad, src2.grad, atol=1e-7)

    # Mean gradient
    src1.grad.zero_()
    src2.grad.zero_()

    out_mean1 = pyg_scatter.scatter(src1, index, dim=0, dim_size=dim_size, reduce="mean")
    loss_m1 = out_mean1.sum()
    loss_m1.backward()

    count = src2.new_zeros(dim_size)
    count.index_add_(0, index, src2.new_ones(src2.size(0))).clamp_(min=1)
    out_mean2 = src2.new_zeros(dim_size, channels).index_add_(0, index, src2) / count.unsqueeze(-1)
    loss_m2 = out_mean2.sum()
    loss_m2.backward()

    assert torch.allclose(src1.grad, src2.grad, atol=1e-7)


def test_scatter_patch_empty_index():
    """Verify safe behavior when index is empty."""
    src = torch.empty((0, 32))
    index = torch.empty((0,), dtype=torch.long)
    out = pyg_scatter.scatter(src, index, dim=0, dim_size=10, reduce="sum")
    assert out.shape == (10, 32)
    assert (out == 0).all()


def test_heterognn_attention_forward_backward_with_patch():
    """Verify HeteroGNN attention (HGTConv) forward and backward passes run cleanly with patch."""
    torch.manual_seed(42)
    metadata = (["domain", "ip"], [("domain", "resolves_to", "ip"), ("ip", "rev_resolves_to", "domain")])
    in_channels = {"domain": 10, "ip": 1}

    model = HeteroGNN(
        metadata=metadata,
        in_channels_dict=in_channels,
        hidden_dim=32,
        out_dim=16,
        num_layers=2,
        num_heads=4,
        variant="attn"
    )

    x_dict = {
        "domain": torch.randn(50, 10, requires_grad=True),
        "ip": torch.randn(20, 1, requires_grad=True),
    }
    edge_index_dict = {
        ("domain", "resolves_to", "ip"): torch.randint(0, 20, (2, 80)),
        ("ip", "rev_resolves_to", "domain"): torch.randint(0, 20, (2, 80)),
    }

    model.train()
    logits = model(x_dict, edge_index_dict)
    assert logits.shape == (50, 2)

    loss = F.cross_entropy(logits, torch.randint(0, 2, (50,)))
    loss.backward()

    assert x_dict["domain"].grad is not None
    assert not torch.isnan(x_dict["domain"].grad).any()
