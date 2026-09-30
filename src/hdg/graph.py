"""
Cumulative, leak-free Heterogeneous Graph construction for Malicious-HDG.
Builds PyG HeteroData with:
- Node types: domain, ip, nameserver, registrar, asn, certificate
- Edges: resolves_to, ns_ip, uses_ns, registered_by, belongs_to_asn, uses_cert + reverse edges
- Train-only feature standardization
- Strictly cumulative edges (no future edges beyond experiment timestamp)
"""

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def compute_shannon_entropy(s: str) -> float:
    """Computes Shannon character entropy."""
    if not s:
        return 0.0
    cnt = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in cnt.values())


def compute_longest_consonant_run(s: str) -> int:
    """Finds longest consecutive run of consonants in domain."""
    vowels_and_special = set("aeiou0123456789-._")
    max_run = 0
    cur_run = 0
    for ch in s.lower():
        if ch.isalpha() and ch not in vowels_and_special:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        else:
            cur_run = 0
    return max_run


def extract_domain_feature_vector(
    row: pd.Series,
    use_lexical: bool = True,
    include_has_subdomain: bool = False
) -> np.ndarray:
    """
    Extracts numerical feature vector for a domain.
    Standardized strictly on training domains later.
    """
    feats: List[float] = [
        float(row.get("no_ip", False)),
        float(row.get("no_rdap", False)),
        float(row.get("no_tls", False)),
        float(row.get("mx_count", 0)),
        float(row.get("has_spf", False)),
        float(row.get("has_dmarc", False)),
        float(row.get("has_dkim", False)),
        float(row.get("dnssec", False)),
        math.log1p(max(0, float(row.get("ttl_a", 0)))),
        math.log1p(max(0, float(row.get("ttl_ns", 0)))),
        float(row.get("tls_cert_count", 0)),
        math.log1p(max(0, float(row.get("cert_valid_len") or 0)))
    ]

    if include_has_subdomain:
        feats.append(float(row.get("has_subdomain", False)))

    if use_lexical:
        d_str = str(row.get("domain") or "")
        d_len = max(len(d_str), 1)
        digits = sum(c.isdigit() for c in d_str)
        vowels = sum(c in "aeiou" for c in d_str.lower())

        feats.extend([
            float(len(d_str)),
            compute_shannon_entropy(d_str),
            float(digits / d_len),
            float(vowels / d_len),
            float(compute_longest_consonant_run(d_str))
        ])

    return np.array(feats, dtype=np.float32)


class TrainOnlyStandardizer:
    """Standardizes feature matrices using statistics fit ONLY on training rows."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, train_idx: np.ndarray) -> "TrainOnlyStandardizer":
        X_train = X[train_idx]
        self.mean = np.nanmean(X_train, axis=0)
        self.std = np.nanstd(X_train, axis=0)
        self.std[self.std == 0] = 1.0
        self.std[np.isnan(self.std)] = 1.0
        self.mean[np.isnan(self.mean)] = 0.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer has not been fit yet!")
        return (X - self.mean) / self.std

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None
        }


def build_hetero_graph(
    df: pd.DataFrame,
    train_indices: Optional[np.ndarray] = None,
    t_max: Optional[str] = None,
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
    include_certificates: bool = True
) -> Tuple[HeteroData, TrainOnlyStandardizer, Dict[str, Dict[str, int]]]:
    """
    Constructs a PyG HeteroData object.
    If t_max is provided, includes strictly domains with t <= t_max (cumulative, no future edges).
    If train_indices is provided, standardizer is fit on train_indices only.
    """
    # Filter cumulative time window
    if t_max is not None:
        valid_df = df[df["t"] <= t_max].copy().reset_index(drop=True)
    else:
        valid_df = df.copy().reset_index(drop=True)

    n_domains = len(valid_df)

    # 1. Feature matrix extraction
    raw_feats = np.array([
        extract_domain_feature_vector(row, use_lexical, include_has_subdomain)
        for _, row in valid_df.iterrows()
    ], dtype=np.float32)

    # Standardize on train indices only (or all if not specified)
    scaler = TrainOnlyStandardizer()
    fit_idx = train_indices if train_indices is not None else np.arange(n_domains)
    scaler.fit(raw_feats, fit_idx)
    norm_feats = scaler.transform(raw_feats)

    # 2. Entity maps
    domain_to_id = {row["domain"]: idx for idx, row in valid_df.iterrows()}
    ip_to_id: Dict[str, int] = {}
    ns_to_id: Dict[str, int] = {}
    reg_to_id: Dict[str, int] = {}
    asn_to_id: Dict[int, int] = {}
    cert_to_id: Dict[str, int] = {}

    # Edges
    d_resolves_ip: List[Tuple[int, int]] = []
    ns_ns_ip: List[Tuple[int, int]] = []
    d_uses_ns: List[Tuple[int, int]] = []
    d_registered_reg: List[Tuple[int, int]] = []
    ip_belongs_asn: List[Tuple[int, int]] = []
    d_uses_cert: List[Tuple[int, int]] = []

    for d_idx, row in valid_df.iterrows():
        # Resolves_to (only A/AAAA IPs)
        res_ips = row.get("resolved_ips")
        if res_ips is not None:
            for ip in res_ips:
                if ip not in ip_to_id:
                    ip_to_id[ip] = len(ip_to_id)
                d_resolves_ip.append((d_idx, ip_to_id[ip]))

        # Uses_ns
        ns_list = row.get("nameservers")
        if ns_list is not None:
            for ns in ns_list:
                if ns not in ns_to_id:
                    ns_to_id[ns] = len(ns_to_id)
                d_uses_ns.append((d_idx, ns_to_id[ns]))

        # Registered_by
        reg = row.get("registrar")
        if reg:
            if reg not in reg_to_id:
                reg_to_id[reg] = len(reg_to_id)
            d_registered_reg.append((d_idx, reg_to_id[reg]))

        # Uses_cert
        if include_certificates:
            cert = row.get("leaf_cert_key")
            if cert:
                if cert not in cert_to_id:
                    cert_to_id[cert] = len(cert_to_id)
                d_uses_cert.append((d_idx, cert_to_id[cert]))

        # IP - ASN and NS - IP
        ip_recs = row.get("ip_records")
        if ip_recs is not None:
            for ip_rec in ip_recs:
                ip_val = ip_rec.get("ip")
                asn_val = ip_rec.get("asn")
                if ip_val and asn_val is not None:
                    if ip_val not in ip_to_id:
                        ip_to_id[ip_val] = len(ip_to_id)
                    if asn_val not in asn_to_id:
                        asn_to_id[asn_val] = len(asn_to_id)
                    ip_belongs_asn.append((ip_to_id[ip_val], asn_to_id[asn_val]))

        ns_ips = row.get("ns_ip_pairs")
        if ns_ips is not None:
            for ns_host, ns_ip in ns_ips:
                if ns_host not in ns_to_id:
                    ns_to_id[ns_host] = len(ns_to_id)
                if ns_ip not in ip_to_id:
                    ip_to_id[ns_ip] = len(ip_to_id)
                ns_ns_ip.append((ns_to_id[ns_host], ip_to_id[ns_ip]))

    # Deduplicate edges
    def to_edge_tensor(pairs: List[Tuple[int, int]]) -> torch.Tensor:
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        unique_pairs = sorted(list(set(pairs)))
        src = [p[0] for p in unique_pairs]
        dst = [p[1] for p in unique_pairs]
        return torch.tensor([src, dst], dtype=torch.long)

    data = HeteroData()
    data["domain"].x = torch.tensor(norm_feats, dtype=torch.float)
    data["domain"].y = torch.tensor(valid_df["label"].to_numpy(), dtype=torch.long)

    # Infrastructure node features: experiment graph degree (log1p)
    def make_infra_features(n_nodes: int, incident_edges: List[Tuple[int, int]], pos: int = 1) -> torch.Tensor:
        if n_nodes == 0:
            return torch.empty((0, 1), dtype=torch.float)
        degs = Counter([p[pos] for p in incident_edges])
        feat = np.array([[math.log1p(degs.get(i, 0))] for i in range(n_nodes)], dtype=np.float32)
        return torch.tensor(feat, dtype=torch.float)

    # IP features
    data["ip"].x = make_infra_features(len(ip_to_id), d_resolves_ip + ns_ns_ip, pos=1)
    # Nameserver features
    data["nameserver"].x = make_infra_features(len(ns_to_id), d_uses_ns, pos=1)
    # Registrar features
    data["registrar"].x = make_infra_features(len(reg_to_id), d_registered_reg, pos=1)
    # ASN features
    data["asn"].x = make_infra_features(len(asn_to_id), ip_belongs_asn, pos=1)
    # Certificate features
    if include_certificates:
        data["certificate"].x = make_infra_features(len(cert_to_id), d_uses_cert, pos=1)

    # Assign forward and reverse edges
    e_res = to_edge_tensor(d_resolves_ip)
    data["domain", "resolves_to", "ip"].edge_index = e_res
    data["ip", "rev_resolves_to", "domain"].edge_index = torch.stack([e_res[1], e_res[0]]) if e_res.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_ns_ip = to_edge_tensor(ns_ns_ip)
    data["nameserver", "ns_ip", "ip"].edge_index = e_ns_ip
    data["ip", "rev_ns_ip", "nameserver"].edge_index = torch.stack([e_ns_ip[1], e_ns_ip[0]]) if e_ns_ip.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_ns = to_edge_tensor(d_uses_ns)
    data["domain", "uses_ns", "nameserver"].edge_index = e_ns
    data["nameserver", "rev_uses_ns", "domain"].edge_index = torch.stack([e_ns[1], e_ns[0]]) if e_ns.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_reg = to_edge_tensor(d_registered_reg)
    data["domain", "registered_by", "registrar"].edge_index = e_reg
    data["registrar", "rev_registered_by", "domain"].edge_index = torch.stack([e_reg[1], e_reg[0]]) if e_reg.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_asn = to_edge_tensor(ip_belongs_asn)
    data["ip", "belongs_to_asn", "asn"].edge_index = e_asn
    data["asn", "rev_belongs_to_asn", "ip"].edge_index = torch.stack([e_asn[1], e_asn[0]]) if e_asn.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    if include_certificates:
        e_cert = to_edge_tensor(d_uses_cert)
        data["domain", "uses_cert", "certificate"].edge_index = e_cert
        data["certificate", "rev_uses_cert", "domain"].edge_index = torch.stack([e_cert[1], e_cert[0]]) if e_cert.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    id_maps = {
        "domain": domain_to_id,
        "ip": ip_to_id,
        "nameserver": ns_to_id,
        "registrar": reg_to_id,
        "asn": {str(k): v for k, v in asn_to_id.items()},
        "certificate": cert_to_id
    }

    return data, scaler, id_maps
