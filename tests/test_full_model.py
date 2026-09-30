"""
Test for legacy FullModel architecture to ensure backwards compatibility.
"""

import os
import sys
from pathlib import Path
import pytest
import torch
from torch_geometric.data import HeteroData

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.full_model import FullModel


def make_fake_snapshot(num_domains: int = 20, num_ip: int = 10) -> HeteroData:
    data = HeteroData()
    data["domain"].x = torch.randn(num_domains, 8)
    data["ip"].x = torch.randn(num_ip, 2)
    data["nameserver"].x = torch.randn(5, 1)
    data["registrar"].x = torch.randn(3, 1)
    data["asn"].x = torch.randn(4, 1)

    resolves_to = torch.randint(0, min(num_domains, num_ip), (2, 15))
    shares_ns = torch.randint(0, 5, (2, 10)).clamp(max=4)
    registered_by = torch.randint(0, 3, (2, 10)).clamp(max=2)
    belongs_asn = torch.randint(0, 4, (2, 8)).clamp(max=3)

    data["domain", "resolves_to", "ip"].edge_index = resolves_to
    data["domain", "shares_nameserver", "nameserver"].edge_index = shares_ns
    data["domain", "registered_by", "registrar"].edge_index = registered_by
    data["ip", "belongs_to_asn", "asn"].edge_index = belongs_asn
    data["ip", "rev_resolves_to", "domain"].edge_index = resolves_to.flip(0)
    data["nameserver", "rev_shares_nameserver", "domain"].edge_index = shares_ns.flip(0)
    data["registrar", "rev_registered_by", "domain"].edge_index = registered_by.flip(0)
    data["asn", "rev_belongs_to_asn", "ip"].edge_index = belongs_asn.flip(0)
    return data


def test_legacy_full_model_forward() -> None:
    snapshots = [make_fake_snapshot() for _ in range(3)]
    domain_snapshot_id = torch.randint(0, 3, (20,))
    model = FullModel(hidden_dim=32)
    out = model(snapshots, domain_snapshot_id)
    assert out.shape == (20, 2)
    assert not torch.isnan(out).any()