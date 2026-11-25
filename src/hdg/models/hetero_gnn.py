"""
Malicious-HDG: Heterogeneous Graph Neural Network Architectures & Adversarial Defenses.

Supported Variants & RFC-003 Extensions:
----------------------------------------
1. 'sage' (Relational GraphSAGE / Gate-HeteroGNN):
   - Sigmoid-Gated Heterogeneous Convolution (`GatedSAGEConvLayer` when `use_gated_agg=True`)
     combining semantic endpoint states and train-visible Inverse Infrastructure Frequency (IIF)
     edge weights without Softmax sum-to-one collapse.
   - Falls back to standard `HeteroConv(SAGEConv)` when `use_gated_agg=False`.
2. 'attn' (Relational Heterogeneous Attention / HGTConv):
   - Multi-head relational Transformer attention over heterogeneous graphs.
3. 'sage_guard' (Relational GNNGuard Defense):
   - Cosine edge pruning (`cos(h_u, h_v) < tau_prune`), IIF modulation, and layer-wise beta memory.
4. 'use_edges=False' (Isolated Domain MLP Baseline):
   - Bypasses graph convolutions while optionally preserving Char-CNN lexical branch.
5. Multi-Modal & Generalization Extensions (RFC-003):
   - Compact `CharCNNBranch` over TLD-stripped `uint8` domain bodies (`[N, 64]`).
   - Neighborhood-completeness gated classifier head (`use_completeness_gate`) fusing
     `h_dom_0`, `h_dom_final`, and availability indicator `m_d = [has_ip, has_ns, has_rdap, has_cert]`.
   - Gradient Reversal Layer (`GradientReversalFn`) + adversarial feed classifier (`num_envs > 1`).
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, HGTConv
from torch_geometric.utils import scatter

from src.hdg.device import apply_memory_efficient_scatter_patch

# Ensure memory-efficient scatter patch is active globally
apply_memory_efficient_scatter_patch()


class GradientReversalFn(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky) for domain/feed-invariant representation learning."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class CharCNNBranch(nn.Module):
    """
    Compact 1D Convolutional Subword Encoder over TLD-stripped ASCII byte tensors `[N, 64]`.
    Total embedding output dim = `out_dim` (default 16). Memory across 931k domains: ~59.6 MB uint8.
    """

    def __init__(self, vocab_size: int = 128, emb_dim: int = 8, num_channels: int = 8, out_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_channels, kernel_size=k, padding=k // 2)
            for k in (3, 4, 5)
        ])
        self.proj = nn.Linear(num_channels * 3, out_dim)

    def forward(self, char_bytes: torch.Tensor) -> torch.Tensor:
        x = self.embedding(char_bytes.long().clamp(0, 127)).transpose(1, 2)  # [N, emb_dim, L]
        pooled = [F.relu(conv(x)).amax(dim=-1) for conv in self.convs]
        return F.relu(self.proj(torch.cat(pooled, dim=-1)))


class GatedSAGEConvLayer(nn.Module):
    """
    Sigmoid-Gated Heterogeneous Message Passing Layer (RFC-003 Section 3.2).

    Formulation:
      g_{uv}^{(r)} = sigmoid( W_{g,src}^{(r)} h_u + W_{g,dst}^{(r)} h_v + W_{g,e}^{(r)} w_IIF(u,v) + b_g )
      h_v^{(l+1)} = W_self h_v + mean_{r} [ (1 / |N_r(v)|) * sum_{u in N_r(v)} g_{uv}^{(r)} * w_IIF * W_r h_u ]

    Uses factorized gate projections so [E, 2H+1] tensors are never materialized (saving >6 GB on 12.7M edges).
    """

    def __init__(self, edge_types: List[Tuple[str, str, str]], hidden_dim: int):
        super().__init__()
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim

        self.lin_src = nn.ModuleDict()
        self.lin_self = nn.ModuleDict()
        self.gate_src = nn.ModuleDict()
        self.gate_dst = nn.ModuleDict()
        self.gate_edge = nn.ModuleDict()

        for edge_type in edge_types:
            key = "__".join(edge_type)
            self.lin_src[key] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.lin_self[key] = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.gate_src[key] = nn.Linear(hidden_dim, 1, bias=False)
            self.gate_dst[key] = nn.Linear(hidden_dim, 1, bias=False)
            self.gate_edge[key] = nn.Linear(1, 1, bias=True)

    def forward(
        self,
        h_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        h_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in h_dict.keys()}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            edge_index = edge_index_dict[edge_type]
            src_t, _, dst_t = edge_type
            if src_t not in h_dict or dst_t not in h_dict:
                continue
            h_src = h_dict[src_t]
            h_dst = h_dict[dst_t]
            num_dst = h_dst.shape[0]
            if edge_index.numel() == 0 or edge_index.shape[1] == 0 or num_dst == 0:
                continue

            key = "__".join(edge_type)
            u = edge_index[0]
            v = edge_index[1]

            w_iif = None
            if edge_weight_dict is not None and edge_type in edge_weight_dict:
                cand_w = edge_weight_dict[edge_type]
                if cand_w is not None and cand_w.shape[0] == edge_index.shape[1]:
                    w_iif = cand_w.to(device=h_dst.device, dtype=h_dst.dtype)
            if w_iif is None:
                w_iif = torch.ones(edge_index.shape[1], dtype=h_dst.dtype, device=h_dst.device)

            # Factorized sigmoid gate over (h_u, h_v, w_IIF)
            g_s = self.gate_src[key](h_src)[u]
            g_d = self.gate_dst[key](h_dst)[v]
            g_e = self.gate_edge[key](w_iif.unsqueeze(-1))
            g_uv = torch.sigmoid((g_s + g_d + g_e).squeeze(-1))

            eff_w = (g_uv * w_iif).unsqueeze(-1)
            src_proj = self.lin_src[key](h_src)
            msg = eff_w * src_proj[u]

            agg_sum = scatter(msg, v, dim=0, dim_size=num_dst, reduce="sum")
            deg_v = scatter(torch.ones_like(w_iif), v, dim=0, dim_size=num_dst, reduce="sum").clamp(min=1.0).unsqueeze(-1)
            out_v = self.lin_self[key](h_dst) + (agg_sum / deg_v)
            h_out[dst_t].append(out_v)

        final_h: Dict[str, torch.Tensor] = {}
        for nt in h_dict.keys():
            if h_out[nt]:
                final_h[nt] = torch.stack(h_out[nt], dim=0).mean(dim=0)
            else:
                final_h[nt] = h_dict[nt]
        return final_h


class GNNGuardLayer(nn.Module):
    """
    Relational GNNGuard Layer for Heterogeneous Graph Adversarial Robustness.
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
        prev_weights_dict: Optional[Dict[str, torch.Tensor]] = None,
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h_out: Dict[str, List[torch.Tensor]] = {nt: [] for nt in h_dict.keys()}
        new_weights_dict: Dict[str, torch.Tensor] = {}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            src_t, _, dst_t = edge_type
            key = "__".join(edge_type)

            h_src = h_dict[src_t]
            h_dst = h_dict[dst_t]
            num_dst = h_dst.shape[0]

            if edge_index.numel() == 0 or edge_index.shape[1] == 0:
                new_weights_dict[key] = torch.empty(0, device=h_dst.device)
                continue

            u = edge_index[0]
            v = edge_index[1]

            x_u = h_src[u]
            x_v = h_dst[v]
            sim = F.cosine_similarity(x_u, x_v, dim=-1, eps=1e-8)
            sim = torch.clamp(sim, min=0.0)

            keep = (sim >= self.prune_threshold).float()
            pruned_sim = sim * keep

            # Incorporate continuous IIF prior if provided
            if edge_weight_dict is not None and edge_type in edge_weight_dict:
                ew = edge_weight_dict[edge_type]
                if ew is not None and ew.shape[0] == edge_index.shape[1]:
                    pruned_sim = pruned_sim * ew.to(device=h_dst.device, dtype=h_dst.dtype)

            denom = scatter(pruned_sim, v, dim=0, dim_size=num_dst, reduce="sum") + 1e-8
            alpha = pruned_sim / denom[v]

            beta_val = torch.sigmoid(self.betas[key])
            prev_w = prev_weights_dict.get(key) if prev_weights_dict is not None else None

            if prev_w is not None and prev_w.shape == alpha.shape:
                cur_w = beta_val * prev_w + (1.0 - beta_val) * alpha
            else:
                cur_w = alpha

            sum_w = scatter(cur_w, v, dim=0, dim_size=num_dst, reduce="sum") + 1e-8
            norm_w = cur_w / sum_w[v]
            new_weights_dict[key] = norm_w

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
    Supports 'sage' (optionally Sigmoid-Gated with IIF), 'attn' (HGTConv), and 'sage_guard' (GNNGuard).
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
        use_edges: bool = True,
        use_gated_agg: bool = True,
        use_char_cnn: bool = False,
        char_cnn_dim: int = 16,
        use_completeness_gate: bool = False,
        num_envs: int = 0,
    ):
        super().__init__()
        apply_memory_efficient_scatter_patch()
        self.variant = variant.lower()
        self.metadata = metadata
        self.in_channels_dict = in_channels_dict
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prune_threshold = prune_threshold
        self.use_edges = use_edges
        self.use_gated_agg = use_gated_agg
        self.use_char_cnn = use_char_cnn
        self.char_cnn_dim = char_cnn_dim
        self.use_completeness_gate = use_completeness_gate
        self.num_envs = num_envs

        node_types, edge_types = metadata
        self.target_node_type = "fqdn" if "fqdn" in node_types else "domain"

        # 1. Feature projection layers for each node type to hidden_dim
        self.proj = nn.ModuleDict()
        for nt in node_types:
            in_dim = in_channels_dict.get(nt, 1)
            self.proj[nt] = nn.Linear(in_dim, hidden_dim)

        # Optional compact Char-CNN branch over domain body bytes
        if self.use_char_cnn:
            self.char_cnn = CharCNNBranch(out_dim=char_cnn_dim)
            self.char_fuse = nn.Linear(hidden_dim + char_cnn_dim, hidden_dim)
        else:
            self.char_cnn = None
            self.char_fuse = None

        # 2. Graph convolution layers
        self.convs = nn.ModuleList()
        if not self.use_edges:
            pass  # Isolated domain MLP (no edge message passing)
        elif self.variant == "sage":
            if self.use_gated_agg:
                for _ in range(num_layers):
                    self.convs.append(GatedSAGEConvLayer(edge_types=edge_types, hidden_dim=hidden_dim))
            else:
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

        # 3. Domain classification head (with optional neighborhood-completeness gate)
        if self.use_completeness_gate:
            self.completeness_gate = nn.Linear(hidden_dim + 4, hidden_dim)
            cls_in_dim = hidden_dim + 4
        else:
            self.completeness_gate = None
            cls_in_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 2)
        )

        # 4. Optional GRL domain-adversarial environment discriminator
        if self.num_envs > 1:
            self.env_classifier = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, self.num_envs),
            )
        else:
            self.env_classifier = None

    @staticmethod
    def _compute_availability_mask(
        num_domains: int,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        target_node_type: str = "domain",
    ) -> torch.Tensor:
        """Computes `[N_target, 4]` availability indicator `[has_ip, has_ns_or_apex, has_rdap_or_apex_reg, has_cert]`."""
        if target_node_type == "fqdn":
            ip_col = torch.zeros(num_domains, dtype=dtype, device=device)
            et_ip = ("fqdn", "resolves_to", "ip")
            if et_ip in edge_index_dict and edge_index_dict[et_ip].numel() > 0:
                ip_col.index_fill_(0, edge_index_dict[et_ip][0], 1.0)

            ns_col = torch.zeros(num_domains, dtype=dtype, device=device)
            reg_col = torch.zeros(num_domains, dtype=dtype, device=device)
            et_sub = ("fqdn", "subdomain_of", "apex_domain")
            if et_sub in edge_index_dict and edge_index_dict[et_sub].numel() > 0:
                f_idx, a_idx = edge_index_dict[et_sub][0], edge_index_dict[et_sub][1]
                et_ns = ("apex_domain", "uses_ns", "nameserver")
                if et_ns in edge_index_dict and edge_index_dict[et_ns].numel() > 0:
                    max_a = max(int(a_idx.max().item()), int(edge_index_dict[et_ns][0].max().item())) + 1
                    a_has_ns = torch.zeros(max_a, dtype=dtype, device=device)
                    a_has_ns.index_fill_(0, edge_index_dict[et_ns][0], 1.0)
                    valid_f = f_idx < num_domains
                    ns_col[f_idx[valid_f]] = a_has_ns[a_idx[valid_f]]
                et_reg = ("apex_domain", "registered_by", "registrar")
                if et_reg in edge_index_dict and edge_index_dict[et_reg].numel() > 0:
                    max_a = max(int(a_idx.max().item()), int(edge_index_dict[et_reg][0].max().item())) + 1
                    a_has_reg = torch.zeros(max_a, dtype=dtype, device=device)
                    a_has_reg.index_fill_(0, edge_index_dict[et_reg][0], 1.0)
                    valid_f = f_idx < num_domains
                    reg_col[f_idx[valid_f]] = a_has_reg[a_idx[valid_f]]

            cert_col = torch.zeros(num_domains, dtype=dtype, device=device)
            et_cert = ("fqdn", "uses_cert", "certificate")
            if et_cert in edge_index_dict and edge_index_dict[et_cert].numel() > 0:
                cert_col.index_fill_(0, edge_index_dict[et_cert][0], 1.0)
            return torch.stack([ip_col, ns_col, reg_col, cert_col], dim=-1)

        rels = [
            ("domain", "resolves_to", "ip"),
            ("domain", "uses_ns", "nameserver"),
            ("domain", "registered_by", "registrar"),
            ("domain", "uses_cert", "certificate"),
        ]
        cols: List[torch.Tensor] = []
        for et in rels:
            col = torch.zeros(num_domains, dtype=dtype, device=device)
            if et in edge_index_dict and edge_index_dict[et].numel() > 0:
                src_nodes = edge_index_dict[et][0]
                col.index_fill_(0, src_nodes, 1.0)
            cols.append(col)
        return torch.stack(cols, dim=-1)

    def _encode_nodes_and_fuse(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        char_bytes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs input projections, CharCNN fusion, GNN layers, and returns `(fused_emb, raw_domain_emb, penult_emb)`."""
        t_nt = self.target_node_type if self.target_node_type in x_dict else ("fqdn" if "fqdn" in x_dict else "domain")
        h_dict: Dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.proj:
                h_dict[nt] = F.relu(self.proj[nt](x))
            else:
                h_dict[nt] = x

        if self.use_char_cnn and self.char_cnn is not None and self.char_fuse is not None:
            h_dom = h_dict[t_nt]
            if char_bytes is not None:
                c_emb = self.char_cnn(char_bytes.to(h_dom.device))
            else:
                c_emb = torch.zeros((h_dom.shape[0], self.char_cnn_dim), dtype=h_dom.dtype, device=h_dom.device)
            h_dict[t_nt] = F.relu(self.char_fuse(torch.cat([h_dom, c_emb], dim=-1)))

        h_dom_0 = h_dict[t_nt]

        if self.use_edges:
            prev_weights_dict: Optional[Dict[str, torch.Tensor]] = None
            for conv in self.convs:
                if self.variant == "sage_guard":
                    h_new, prev_weights_dict = conv(
                        h_dict, edge_index_dict, prev_weights_dict, edge_weight_dict=edge_weight_dict
                    )
                elif self.variant == "sage" and self.use_gated_agg:
                    h_new = conv(h_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
                else:
                    h_new = conv(h_dict, edge_index_dict)

                for nt in h_dict.keys():
                    if nt in h_new and h_new[nt] is not None:
                        h_dict[nt] = F.relu(h_new[nt] + h_dict[nt])
                        h_dict[nt] = F.dropout(h_dict[nt], p=self.dropout, training=self.training)

        domain_emb = h_dict[t_nt]

        if self.use_completeness_gate and self.completeness_gate is not None:
            m_d = self._compute_availability_mask(
                num_domains=domain_emb.shape[0],
                edge_index_dict=edge_index_dict,
                device=domain_emb.device,
                dtype=domain_emb.dtype,
                target_node_type=t_nt,
            )
            comp_gate = torch.sigmoid(self.completeness_gate(torch.cat([h_dom_0, m_d], dim=-1)))
            fused_emb = comp_gate * domain_emb + (1.0 - comp_gate) * h_dom_0
            cls_in = torch.cat([fused_emb, m_d], dim=-1)
        else:
            cls_in = domain_emb

        h_penult = F.relu(self.classifier[0](cls_in))
        return h_penult, domain_emb, cls_in

    def forward_with_aux(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        char_bytes: Optional[torch.Tensor] = None,
        grl_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning `(logits, domain_emb, env_logits)`.
        """
        h_penult, domain_emb, _ = self._encode_nodes_and_fuse(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            char_bytes=char_bytes,
        )
        h_drop = self.classifier[2](h_penult)
        logits = self.classifier[3](h_drop)

        env_logits: Optional[torch.Tensor] = None
        if self.env_classifier is not None:
            rev_emb = GradientReversalFn.apply(domain_emb, grl_lambda)
            env_logits = self.env_classifier(rev_emb)

        return logits, domain_emb, env_logits

    def get_penultimate_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        char_bytes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns `(logits, penultimate_emb)` where `penultimate_emb` has shape `[N, out_dim]` (RFC-002 Stacker)."""
        h_penult, _, _ = self._encode_nodes_and_fuse(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            char_bytes=char_bytes,
        )
        logits = self.classifier[3](self.classifier[2](h_penult))
        return logits, h_penult

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_weight_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
        char_bytes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through heterogeneous network.
        Returns: Logits for domain nodes of shape `[N_domains, 2]`.
        """
        logits, _, _ = self.forward_with_aux(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            char_bytes=char_bytes,
            grl_lambda=0.0,
        )
        return logits

    def get_model_config(self) -> Dict[str, Any]:
        """Returns comprehensive architecture config for reliable checkpoint serialization."""
        return {
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "prune_threshold": self.prune_threshold,
            "use_edges": self.use_edges,
            "use_gated_agg": self.use_gated_agg,
            "use_char_cnn": self.use_char_cnn,
            "char_cnn_dim": self.char_cnn_dim,
            "use_completeness_gate": self.use_completeness_gate,
            "num_envs": self.num_envs,
            "target_node_type": self.target_node_type,
        }
