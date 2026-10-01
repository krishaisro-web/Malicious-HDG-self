import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

NODE_TYPES = ["domain", "ip", "nameserver", "registrar", "asn"]
IN_DIMS = {"domain": 8, "ip": 2, "nameserver": 1, "registrar": 1, "asn": 1}

ALL_RELATIONS = {
    ("domain", "resolves_to", "ip"),
    ("domain", "shares_nameserver", "nameserver"),
    ("domain", "registered_by", "registrar"),
    ("ip", "belongs_to_asn", "asn"),
    ("ip", "rev_resolves_to", "domain"),
    ("nameserver", "rev_shares_nameserver", "domain"),
    ("registrar", "rev_registered_by", "domain"),
    ("asn", "rev_belongs_to_asn", "ip"),
}

class FeatureProjection(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.ModuleDict({
            nt: nn.Linear(IN_DIMS[nt], hidden_dim) for nt in NODE_TYPES
        })

    def forward(self, x_dict):
        return {nt: torch.relu(self.proj[nt](x)) for nt, x in x_dict.items()}

def make_hetero_layer(hidden_dim, relations=None):
    if relations is None:
        relations = ALL_RELATIONS
    conv_dict = {rel: SAGEConv((-1, -1), hidden_dim) for rel in relations}
    return HeteroConv(conv_dict, aggr="mean")
