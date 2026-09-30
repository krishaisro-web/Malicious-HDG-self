"""
Heterogeneous Graph Neural Network classifiers for Malicious-HDG.
Provides two single-graph forward architectures:
1. 'sage': Relational GraphSAGE using HeteroConv + SAGEConv.
2. 'attn': Relational Heterogeneous Attention using PyG HGTConv (SHetGCN-inspired, not a reproduction).
CPU-friendly with linear projection of heterogeneous input features.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Classifier for domain maliciousness detection.
    Variant 'sage': HeteroConv with SAGEConv message passing.
    Variant 'attn': Relational Heterogeneous Attention (HGTConv), inspired by SHetGCN.
    """

    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        in_channels_dict: Dict[str, int],
        hidden_dim: int = 64,
        out_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        variant: str = "sage"
    ):
        super().__init__()
        self.variant = variant.lower()
        self.metadata = metadata
        self.num_layers = num_layers
        self.dropout = dropout

        node_types, edge_types = metadata

        # 1. Feature projection layers for each node type to hidden_dim
        self.proj = nn.ModuleDict()
        for nt in node_types:
            in_dim = in_channels_dict.get(nt, 1)
            self.proj[nt] = nn.Linear(in_dim, hidden_dim)

        # 2. Graph convolution layers
        self.convs = nn.ModuleList()
        if self.variant == "sage":
            for _ in range(num_layers):
                conv_dict = {
                    edge_type: SAGEConv(hidden_dim, hidden_dim)
                    for edge_type in edge_types
                }
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))
        elif self.variant == "attn":
            # SHetGCN-inspired attention mechanism over heterogeneous relations
            for _ in range(num_layers):
                self.convs.append(
                    HGTConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        metadata=metadata,
                        heads=num_heads
                    )
                )
        else:
            raise ValueError(f"Unknown GNN variant: '{variant}'. Supported: 'sage', 'attn'.")

        # 3. Domain classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 2)
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through heterogeneous network.
        Returns: Logits for domain nodes of shape [N_domains, 2].
        """
        # Linear projection
        h_dict = {}
        for nt, x in x_dict.items():
            if nt in self.proj:
                h_dict[nt] = F.relu(self.proj[nt](x))
            else:
                h_dict[nt] = x

        # Message passing layers
        for i, conv in enumerate(self.convs):
            h_new = conv(h_dict, edge_index_dict)
            for nt in h_dict.keys():
                if nt in h_new:
                    # Residual connection + activation + dropout
                    h_dict[nt] = F.relu(h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)

        # Classify domain nodes
        domain_emb = h_dict["domain"]
        logits = self.classifier(domain_emb)
        return logits
