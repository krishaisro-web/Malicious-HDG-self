"""
Heterogeneous Graph Neural Network classifiers for Malicious-HDG.
Provides three single-graph forward architectures:
1. 'sage': Relational GraphSAGE using HeteroConv + SAGEConv.
2. 'attn': Relational Heterogeneous Attention using PyG HGTConv (SHetGCN-inspired).
3. 'sage_guard': GNNGuard defense (Zhang & Zitnik, NeurIPS 2020) with cosine edge pruning,
   destination row-normalization, layer-wise beta memory, and weighted scatter aggregation.
4. 'use_edges=False': MLP on domain features only (mlp_no_edges ablation).
CPU-friendly with linear projection of heterogeneous input features.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv
from torch_geometric.utils import scatter


class GNNGuardLayer(nn.Module):
    """
    Relational GNNGuard layer for heterogeneous graphs.
    Computes cosine similarity between hidden representations of edge endpoints,
    prunes edges below prune_threshold, row-normalizes weights per destination node,
    applies layer-wise memory with learnable beta, and computes weighted message aggregation.
    """

    def __init__(
        self,
        edge_types: List[Tuple[str, str, str]],
        hidden_dim: int,
        prune_threshold: float = 0.1
    ):
        super().__init__()
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.prune_threshold = prune_threshold

        self.betas = nn.ParameterDict()
        self.lin_src = nn.ModuleDict()
        self.lin_self = nn.ModuleDict()

        for edge_type in edge_types:
            key = "__".join(edge_type)
            self.betas[key] = nn.Parameter(torch.zeros(1))
            self.lin_src[key] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.lin_self[key] = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        prev_weights_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in h_dict.keys()}
        new_weights_dict: Dict[str, torch.Tensor] = {}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            src_t, rel, dst_t = edge_type
            key = "__".join(edge_type)

            h_src = h_dict[src_t]
            h_dst = h_dict[dst_t]
            num_dst = h_dst.shape[0]

            if edge_index.numel() == 0 or edge_index.shape[1] == 0:
                new_weights_dict[key] = torch.empty(0, device=h_dst.device)
                continue

            u = edge_index[0]
            v = edge_index[1]

            # 1. Cosine similarity between projected endpoint vectors
            x_u = h_src[u]
            x_v = h_dst[v]
            sim = F.cosine_similarity(x_u, x_v, dim=-1, eps=1e-8)
            sim = torch.clamp(sim, min=0.0)

            # 2. Prune edges below threshold
            keep = (sim >= self.prune_threshold).float()
            pruned_sim = sim * keep

            # 3. Row-normalization over destination node's incoming edges
            denom = scatter(pruned_sim, v, dim=0, dim_size=num_dst, reduce="sum") + 1e-8
            alpha = pruned_sim / denom[v]

            # 4. Layer-wise memory: weight_k = beta * weight_{k-1} + (1 - beta) * alpha
            beta_val = torch.sigmoid(self.betas[key])
            prev_w = prev_weights_dict.get(key) if prev_weights_dict is not None else None

            if prev_w is not None and prev_w.shape == alpha.shape:
                cur_w = beta_val * prev_w + (1.0 - beta_val) * alpha
            else:
                cur_w = alpha

            # Normalize weights to sum to 1 per destination node
            sum_w = scatter(cur_w, v, dim=0, dim_size=num_dst, reduce="sum") + 1e-8
            norm_w = cur_w / sum_w[v]
            new_weights_dict[key] = norm_w

            # 5. Weighted message aggregation (mean SAGE-style)
            msg = norm_w.unsqueeze(-1) * self.lin_src[key](x_u)
            agg = scatter(msg, v, dim=0, dim_size=num_dst, reduce="sum")
            out_v = self.lin_self[key](h_dst) + agg
            h_out[dst_t].append(out_v)

        final_h = {}
        for nt in h_dict.keys():
            if h_out[nt]:
                final_h[nt] = torch.stack(h_out[nt], dim=0).mean(dim=0)
            else:
                final_h[nt] = h_dict[nt]

        return final_h, new_weights_dict


class HeteroGNN(nn.Module):
    """
    Heterogeneous Graph Classifier for domain maliciousness detection.
    Variant 'sage': HeteroConv with SAGEConv message passing.
    Variant 'attn': Relational Heterogeneous Attention (HGTConv), inspired by SHetGCN.
    Variant 'sage_guard': GNNGuard defense against structural evasion.
    Supports use_edges=False for isolated domain MLP baseline.
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
        variant: str = "sage",
        prune_threshold: float = 0.1,
        use_edges: bool = True
    ):
        super().__init__()
        self.variant = variant.lower()
        self.metadata = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edges = use_edges

        node_types, edge_types = metadata

        # 1. Feature projection layers for each node type to hidden_dim
        self.proj = nn.ModuleDict()
        for nt in node_types:
            in_dim = in_channels_dict.get(nt, 1)
            self.proj[nt] = nn.Linear(in_dim, hidden_dim)

        # 2. Graph convolution layers
        self.convs = nn.ModuleList()
        if not self.use_edges:
            pass  # Isolated domain MLP (no edge message passing)
        elif self.variant == "sage":
            for _ in range(num_layers):
                conv_dict = {
                    edge_type: SAGEConv(hidden_dim, hidden_dim)
                    for edge_type in edge_types
                }
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))
        elif self.variant == "attn":
            for _ in range(num_layers):
                self.convs.append(
                    HGTConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        metadata=metadata,
                        heads=num_heads
                    )
                )
        elif self.variant == "sage_guard":
            for _ in range(num_layers):
                self.convs.append(
                    GNNGuardLayer(
                        edge_types=edge_types,
                        hidden_dim=hidden_dim,
                        prune_threshold=prune_threshold
                    )
                )
        else:
            raise ValueError(f"Unknown GNN variant: '{variant}'. Supported: 'sage', 'attn', 'sage_guard'.")

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
        Guards against None outputs in forward and applies residual connections.
        Returns: Logits for domain nodes of shape [N_domains, 2].
        """
        # Linear projection
        h_dict = {}
        for nt, x in x_dict.items():
            if nt in self.proj:
                h_dict[nt] = F.relu(self.proj[nt](x))
            else:
                h_dict[nt] = x

        # Skip graph convolution if use_edges=False (mlp_no_edges)
        if not self.use_edges:
            domain_emb = h_dict["domain"]
            return self.classifier(domain_emb)

        # Message passing layers
        prev_weights_dict: Optional[Dict[str, torch.Tensor]] = None
        for i, conv in enumerate(self.convs):
            if self.variant == "sage_guard":
                h_new, prev_weights_dict = conv(h_dict, edge_index_dict, prev_weights_dict)
            else:
                h_new = conv(h_dict, edge_index_dict)

            for nt in h_dict.keys():
                # Guard against None outputs in forward (e.g. HGTConv on inactive node types)
                if nt in h_new and h_new[nt] is not None:
                    h_dict[nt] = F.relu(h_new[nt] + h_dict[nt])
                    h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)

        domain_emb = h_dict["domain"]
        logits = self.classifier(domain_emb)
        return logits
