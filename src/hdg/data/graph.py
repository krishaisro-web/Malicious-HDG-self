"""
Malicious-HDG: Cumulative, Leak-Free Heterogeneous Graph Construction & Scaling.

Architectural Overview:
-----------------------
This module constructs and manages the heterogeneous domain-infrastructure graph for
Malicious-HDG: Heterogeneous Graph Deep Learning for Proactive DNS Cyber Defense.

Graph Topology Specification (Full 9.3 Lakh Dataset Scale):
-----------------------------------------------------------
1. Node Types:
   - `domain`: 930,000 nodes (22 features: 12 structural/security + 1 subdomain + 9 lexical/DGA signals)
   - `ip`: ~1,470,000 nodes (10 features: incident edge degree + topological embeddings)
   - `nameserver`: ~670,000 nodes (10 features: incident domain degree + topological embeddings)
   - `asn`: ~167,000 nodes (10 features: autonomous system aggregation)
   - `registrar`: ~350,000 nodes (10 features: entity registration context)
   - `certificate`: ~350,000 nodes (10 features: leaf SSL/TLS certificate linkage)
   - `prefix` (optional): BGP route aggregation nodes (IP in prefix -> prefix belongs to ASN)
   Total Nodes: ~3,587,000 heterogeneous nodes.

2. Canonical Edge Relations (6 Bidirectional Canonical Types):
   - `(domain, resolves_to, ip)` & `(ip, rev_resolves_to, domain)`
   - `(domain, uses_ns, nameserver)` & `(nameserver, rev_uses_ns, domain)`
   - `(nameserver, ns_ip, ip)` & `(ip, rev_ns_ip, nameserver)`
   - `(domain, registered_by, registrar)` & `(registrar, rev_registered_by, domain)`
   - `(ip, belongs_to_asn, asn)` & `(asn, rev_belongs_to_asn, ip)`
   - `(domain, uses_cert, certificate)` & `(certificate, rev_uses_cert, domain)`
   Total Edges: ~12,700,000 relational links.

Design Principles & Leakage Prevention:
---------------------------------------
- Two-Stage Decoupled Construction:
    1. Base Graph Construction (`build_base_graph`): Constructs the raw, unscaled graph topology
       once and caches it. No data scalers or label statistics are computed here to prevent leakage.
    2. Split-Specific Standardization (`apply_split_scaling`): Applies z-score feature scaling fit
       strictly on training domains and train-incident infrastructure nodes (z = (x - mu_train) / sigma_train).
- High-Performance Vectorization:
    Iterating 930k rows via DataFrame.iterrows() is replaced with direct column-list indexing,
    reducing build time from ~30 minutes to ~20 seconds while avoiding 930k pd.Series allocations.
- True Topology Isolation:
    For temporal and bipartite group splits, edges incident to future (test) domains are pruned
    at tensor level (~is_domain_test[edge_tensor[0]]), enforcing genuine historical/group isolation.
"""

from collections import Counter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.hdg.data.features import (
    INFRA_FEATURE_SCHEMAS,
    INFRA_TOTAL_DIMS,
    build_apex_intrinsic_features,
    build_asn_intrinsic_features,
    build_certificate_intrinsic_features,
    build_ip_intrinsic_features,
    build_nameserver_intrinsic_features,
    build_prefix_intrinsic_features,
    build_registrar_intrinsic_features,
    compute_longest_consonant_run,
    compute_shannon_entropy,
    encode_domain_char_bytes,
    extract_domain_body,
    extract_domain_feature_vector,
    extract_domain_features_vectorized,
    get_domain_feature_names,
    get_feature_indices,
    get_primary_domain_node_type,
    is_multitenant_public_suffix_apex,
)

logger = logging.getLogger(__name__)

# Re-export feature utilities for backwards compatibility
__all__ = [
    "compute_shannon_entropy",
    "compute_longest_consonant_run",
    "extract_domain_feature_vector",
    "extract_domain_features_vectorized",
    "get_feature_indices",
    "get_domain_feature_names",
    "TrainOnlyStandardizer",
    "build_base_graph",
    "apply_split_scaling",
    "build_hetero_graph",
    "save_graph_cohort_bundle",
    "load_aligned_graph_cohort",
    "build_temporal_graph_pair",
]


class TrainOnlyStandardizer:
    """Standardizes feature matrices using statistics fit ONLY on training rows."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, train_idx: np.ndarray) -> "TrainOnlyStandardizer":
        if len(train_idx) == 0:
            self.mean = np.zeros(X.shape[1], dtype=np.float32)
            self.std = np.ones(X.shape[1], dtype=np.float32)
            return self
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


def build_base_graph(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    t_max: Optional[str] = None
) -> HeteroData:
    """
    Constructs and caches the raw, unscaled HeteroData base graph topology.
    
    Contains unstandardized raw feature matrices, all relational edges, and entity ID mappings.
    Constructed once per cohort; no data scalers or label statistics are fit on full data here,
    guaranteeing zero data leakage before split assignment.

    Algorithmic Workflow:
    1. Time Window Filtering: Optionally subsets records up to historical cutoff `t_max`.
    2. Vectorized Feature Extraction: Computes unscaled domain feature matrices in C/NumPy.
    3. Flat Column Extractions: Pre-extracts DataFrame columns as native Python lists to eliminate
       930k slow pandas Series allocations (~30x speedup).
    4. Entity ID Dictionary Mapping: Assigns contiguous 0-indexed integers to unique infrastructure nodes.
    5. Deduplicated Edge Tensor Generation: Converts raw (src, dst) pairs to sorted (2, E) int64 PyTorch tensors.
    6. Bidirectional Assignment: Populates both forward and canonical reverse edges on HeteroData.
    7. Incident Pair Caching: Preserves raw domain-incident pairs for fast split degree standardizations.

    Args:
        df: Input domain dataframe containing raw features, labels, and entity arrays.
        config: Application configuration dictionary (for graph/cert_key_mode, BGP prefix, etc.).
        t_max: Optional ISO timestamp string representing cumulative temporal cutoff.

    Returns:
        Unscaled PyTorch Geometric HeteroData instance representing the full graph topology.
    """
    if config is None:
        config = {}

    g_cfg = config.get("graph", {})
    p_cfg = config.get("parsing", {})
    cert_key_mode = g_cfg.get("cert_key_mode", "cn")
    include_bgp_prefix = g_cfg.get("include_bgp_prefix", False)
    include_sub = p_cfg.get("include_has_subdomain", False)
    hierarchical_subdomains = bool(
        g_cfg.get("hierarchical_subdomains", False) or p_cfg.get("preserve_subdomains", False)
    )
    primary_nt = "fqdn" if hierarchical_subdomains else "domain"

    # 1. Filter cumulative time window if requested
    if t_max is not None:
        valid_df = df[df["t"] <= t_max].copy().reset_index(drop=True)
    else:
        valid_df = df.copy().reset_index(drop=True)

    n_domains = len(valid_df)

    # 2. Vectorized domain feature extraction (raw, unscaled) + uint8 body bytes
    raw_domain_feats = extract_domain_features_vectorized(
        valid_df,
        use_lexical=True,
        include_has_subdomain=include_sub
    )
    if "domain_body" in valid_df.columns:
        raw_bodies = [
            str(b) if (b is not None and not (isinstance(b, float) and np.isnan(b)) and str(b))
            else extract_domain_body(str(d))
            for b, d in zip(valid_df["domain_body"].tolist(), valid_df["domain"].tolist())
        ]
    else:
        raw_bodies = [extract_domain_body(str(d)) for d in valid_df["domain"].astype(str).tolist()]
    domain_char_bytes = encode_domain_char_bytes(raw_bodies, max_len=64)

    # 3. Entity ID mapping tables (maps external string identifier to contiguous 0-indexed node ID)
    domain_to_id = {dom: idx for idx, dom in enumerate(valid_df["domain"])}
    apex_to_id: Dict[str, int] = {}
    ip_to_id: Dict[str, int] = {}
    ns_to_id: Dict[str, int] = {}
    reg_to_id: Dict[str, int] = {}
    asn_to_id: Dict[int, int] = {}
    cert_to_id: Dict[str, int] = {}
    prefix_to_id: Dict[str, int] = {}

    # Deduplicated intrinsic node metadata dictionaries (strictly split-invariant, zero label leakage)
    apex_meta: Dict[str, Dict[str, Any]] = {}
    ip_meta: Dict[str, Dict[str, Any]] = {}
    ns_meta: Dict[str, Dict[str, Any]] = {}
    asn_meta: Dict[int, Dict[str, Any]] = {}
    reg_meta: Dict[str, Dict[str, Any]] = {}
    cert_meta: Dict[str, Dict[str, Any]] = {}

    # Accumulators for relational edge pairs: List of (source_node_id, destination_node_id)
    fqdn_subdomain_apex: List[Tuple[int, int]] = []
    d_resolves_ip: List[Tuple[int, int]] = []
    ns_ns_ip: List[Tuple[int, int]] = []
    d_uses_ns: List[Tuple[int, int]] = []
    d_registered_reg: List[Tuple[int, int]] = []
    ip_belongs_asn: List[Tuple[int, int]] = []
    d_uses_cert: List[Tuple[int, int]] = []
    ip_in_prefix: List[Tuple[int, int]] = []
    prefix_belongs_asn: List[Tuple[int, int]] = []

    # Map certificate key based on mode ("cn" leaf common name vs "coissue" co-issuance hash)
    cert_col = "leaf_cert_key_coissue" if cert_key_mode == "coissue" else "leaf_cert_key_cn"

    # Optimization: Pre-extract DataFrame columns to flat lists to eliminate 930k slow pd.Series allocations
    col_domains = valid_df["domain"].astype(str).tolist()
    if "apex_domain" in valid_df.columns:
        col_apex = valid_df["apex_domain"].fillna(valid_df["domain"]).astype(str).str.lower().tolist()
    elif "e2LD" in valid_df.columns:
        col_apex = valid_df["e2LD"].fillna(valid_df["domain"]).astype(str).str.lower().tolist()
    else:
        col_apex = [d.lower() for d in col_domains]
    col_t_reg = valid_df["t_reg"].tolist() if "t_reg" in valid_df.columns else [None] * n_domains
    col_t_eff = (
        valid_df["t_eval"].fillna(valid_df["t"]).tolist()
        if ("t_eval" in valid_df.columns and "t" in valid_df.columns)
        else (valid_df["t"].tolist() if "t" in valid_df.columns else [None] * n_domains)
    )
    col_no_rdap = valid_df["no_rdap"].tolist() if "no_rdap" in valid_df.columns else [True] * n_domains

    col_resolved_ips = valid_df["resolved_ips"].tolist() if "resolved_ips" in valid_df.columns else [None] * n_domains
    col_nameservers = valid_df["nameservers"].tolist() if "nameservers" in valid_df.columns else [None] * n_domains
    col_registrar = valid_df["registrar"].tolist() if "registrar" in valid_df.columns else [None] * n_domains
    col_reg_whois = valid_df["reg_has_whois"].tolist() if "reg_has_whois" in valid_df.columns else [False] * n_domains
    col_reg_email = valid_df["reg_has_email"].tolist() if "reg_has_email" in valid_df.columns else [False] * n_domains
    col_reg_url = valid_df["reg_has_url"].tolist() if "reg_has_url" in valid_df.columns else [False] * n_domains
    col_reg_tel = valid_df["reg_has_tel"].tolist() if "reg_has_tel" in valid_df.columns else [False] * n_domains

    if cert_col in valid_df.columns:
        col_cert = valid_df[cert_col].tolist()
    elif "leaf_cert_key" in valid_df.columns:
        col_cert = valid_df["leaf_cert_key"].tolist()
    else:
        col_cert = [None] * n_domains
    col_cert_vlen = valid_df["cert_valid_len"].tolist() if "cert_valid_len" in valid_df.columns else [None] * n_domains
    col_cert_ext = valid_df["cert_ext_count"].tolist() if "cert_ext_count" in valid_df.columns else [0] * n_domains
    col_cert_org = valid_df["cert_org"].tolist() if "cert_org" in valid_df.columns else [None] * n_domains
    col_cert_self = valid_df["cert_is_self_signed"].tolist() if "cert_is_self_signed" in valid_df.columns else [False] * n_domains

    col_ip_records = valid_df["ip_records"].tolist() if "ip_records" in valid_df.columns else [None] * n_domains
    col_ns_ip_pairs = valid_df["ns_ip_pairs"].tolist() if "ns_ip_pairs" in valid_df.columns else [None] * n_domains
    col_mx_ip_pairs = valid_df["mx_ip_pairs"].tolist() if "mx_ip_pairs" in valid_df.columns else [None] * n_domains
    col_cname_targets = valid_df["cname_targets"].tolist() if "cname_targets" in valid_df.columns else [None] * n_domains

    domain_has_resolved_ip = [False] * n_domains
    domain_mx_ips: Dict[int, List[str]] = {}

    # Precompute domain ages safely in bulk if hierarchical mode needs apex_domain ages
    apex_ages = np.zeros(n_domains, dtype=np.float32)
    apex_has_age = np.zeros(n_domains, dtype=bool)
    if hierarchical_subdomains and n_domains > 0:
        eff_s = pd.to_datetime(pd.Series(col_t_eff), errors="coerce", utc=True)
        reg_s = pd.to_datetime(pd.Series(col_t_reg), errors="coerce", utc=True)
        v_mask = eff_s.notna() & reg_s.notna()
        if v_mask.any():
            d_days = np.maximum(0.0, (eff_s[v_mask] - reg_s[v_mask]).dt.total_seconds().to_numpy() / 86400.0)
            apex_ages[v_mask.to_numpy()] = d_days.astype(np.float32)
            apex_has_age[v_mask.to_numpy()] = True

    # 4. Construct entity mappings and populate edge tuples across all domains
    for d_idx in range(n_domains):
        # Optional RFC-001 Apex tier mapping
        apex_id_for_row = d_idx
        if hierarchical_subdomains:
            apex_str = col_apex[d_idx]
            if apex_str not in apex_to_id:
                apex_to_id[apex_str] = len(apex_to_id)
                apex_meta[apex_str] = {
                    "subdomain_fanout_count": 0,
                    "is_public_suffix": is_multitenant_public_suffix_apex(apex_str),
                    "no_rdap": bool(col_no_rdap[d_idx]),
                    "reg_age_days": float(apex_ages[d_idx]),
                    "has_reg_age": bool(apex_has_age[d_idx]),
                    "ns_count": len(col_nameservers[d_idx]) if isinstance(col_nameservers[d_idx], (list, np.ndarray)) else 0,
                    "has_whois": bool(col_reg_whois[d_idx]),
                    "has_email": bool(col_reg_email[d_idx]),
                    "tld": apex_str.rsplit(".", 1)[-1] if "." in apex_str else "",
                }
            apex_meta[apex_str]["subdomain_fanout_count"] += 1
            apex_id_for_row = apex_to_id[apex_str]
            fqdn_subdomain_apex.append((d_idx, apex_id_for_row))

        zone_src_idx = apex_id_for_row if hierarchical_subdomains else d_idx

        # Relation: (domain|fqdn, resolves_to, ip)
        res_ips = col_resolved_ips[d_idx]
        if res_ips is not None and isinstance(res_ips, (list, np.ndarray)) and len(res_ips) > 0:
            domain_has_resolved_ip[d_idx] = True
            for ip in res_ips:
                if ip not in ip_to_id:
                    ip_to_id[ip] = len(ip_to_id)
                d_resolves_ip.append((d_idx, ip_to_id[ip]))

        # Relation: (domain|apex_domain, uses_ns, nameserver)
        ns_list = col_nameservers[d_idx]
        if ns_list is not None and isinstance(ns_list, (list, np.ndarray)):
            for ns in ns_list:
                if ns not in ns_to_id:
                    ns_to_id[ns] = len(ns_to_id)
                d_uses_ns.append((zone_src_idx, ns_to_id[ns]))

        # Relation: (domain|apex_domain, registered_by, registrar)
        reg = col_registrar[d_idx]
        if reg and (not isinstance(reg, float) or not np.isnan(reg)):
            if reg not in reg_to_id:
                reg_to_id[reg] = len(reg_to_id)
            if reg not in reg_meta:
                reg_meta[reg] = {
                    "has_whois": bool(col_reg_whois[d_idx]),
                    "has_email": bool(col_reg_email[d_idx]),
                    "has_url": bool(col_reg_url[d_idx]),
                    "has_tel": bool(col_reg_tel[d_idx]),
                }
            d_registered_reg.append((zone_src_idx, reg_to_id[reg]))

        # Relation: (domain|fqdn, uses_cert, certificate)
        cert = col_cert[d_idx]
        if cert and (not isinstance(cert, float) or not np.isnan(cert)):
            if cert not in cert_to_id:
                cert_to_id[cert] = len(cert_to_id)
            if cert not in cert_meta:
                v_l = col_cert_vlen[d_idx]
                c_org = col_cert_org[d_idx]
                cert_meta[cert] = {
                    "valid_len": None if (v_l is None or (isinstance(v_l, float) and np.isnan(v_l))) else v_l,
                    "ext_count": 0 if (col_cert_ext[d_idx] is None or (isinstance(col_cert_ext[d_idx], float) and np.isnan(col_cert_ext[d_idx]))) else col_cert_ext[d_idx],
                    "org": None if (c_org is None or (isinstance(c_org, float) and np.isnan(c_org))) else c_org,
                    "is_self_signed": bool(col_cert_self[d_idx]),
                }
            d_uses_cert.append((d_idx, cert_to_id[cert]))

        # Infrastructure Relations: (ip, belongs_to_asn, asn) and optional BGP prefix
        ip_recs = col_ip_records[d_idx]
        if ip_recs is not None and isinstance(ip_recs, (list, np.ndarray)):
            for ip_rec in ip_recs:
                if not isinstance(ip_rec, dict):
                    continue
                ip_val = ip_rec.get("ip")
                asn_val = ip_rec.get("asn")
                prefix_val = ip_rec.get("prefix")

                if ip_val:
                    if ip_val not in ip_to_id:
                        ip_to_id[ip_val] = len(ip_to_id)
                    if ip_val not in ip_meta:
                        ip_meta[ip_val] = {
                            "country": ip_rec.get("country"),
                            "country_code": ip_rec.get("country_code"),
                            "latitude": ip_rec.get("latitude"),
                            "longitude": ip_rec.get("longitude"),
                            "prefix_len": ip_rec.get("prefix_len"),
                            "isp": ip_rec.get("isp"),
                            "org": ip_rec.get("org") or ip_rec.get("asn_org"),
                        }

                if ip_val and asn_val is not None:
                    asn_int = int(asn_val)
                    if asn_int not in asn_to_id:
                        asn_to_id[asn_int] = len(asn_to_id)
                    if asn_int not in asn_meta:
                        asn_meta[asn_int] = {
                            "as_org": ip_rec.get("asn_org"),
                            "prefixes": set(),
                            "country": ip_rec.get("country_code") or ip_rec.get("country"),
                        }
                    else:
                        if not asn_meta[asn_int].get("as_org") and ip_rec.get("asn_org"):
                            asn_meta[asn_int]["as_org"] = ip_rec.get("asn_org")
                    if prefix_val:
                        asn_meta[asn_int]["prefixes"].add(str(prefix_val))

                    ip_belongs_asn.append((ip_to_id[ip_val], asn_to_id[asn_int]))

                    if include_bgp_prefix and prefix_val:
                        if prefix_val not in prefix_to_id:
                            prefix_to_id[prefix_val] = len(prefix_to_id)
                        ip_in_prefix.append((ip_to_id[ip_val], prefix_to_id[prefix_val]))
                        prefix_belongs_asn.append((prefix_to_id[prefix_val], asn_to_id[asn_int]))

        # Infrastructure Relation: (nameserver, ns_ip, ip)
        ns_ips = col_ns_ip_pairs[d_idx]
        if ns_ips is not None and isinstance(ns_ips, (list, np.ndarray)):
            for ns_host, ns_ip in ns_ips:
                if ns_host not in ns_to_id:
                    ns_to_id[ns_host] = len(ns_to_id)
                if ns_ip not in ip_to_id:
                    ip_to_id[ns_ip] = len(ip_to_id)
                ns_ns_ip.append((ns_to_id[ns_host], ip_to_id[ns_ip]))
                meta_ns = ns_meta.setdefault(
                    ns_host, {"glue_ips": set(), "glue_asns": set(), "glue_countries": set(), "glue_orgs": set()}
                )
                meta_ns["glue_ips"].add(ns_ip)

        # Dormant MX glue IPs for no_ip cold-start bridging
        mx_ips = col_mx_ip_pairs[d_idx]
        if not domain_has_resolved_ip[d_idx] and mx_ips is not None and isinstance(mx_ips, (list, np.ndarray)):
            for _, mx_ip in mx_ips:
                if mx_ip:
                    if mx_ip not in ip_to_id:
                        ip_to_id[mx_ip] = len(ip_to_id)
                    domain_mx_ips.setdefault(d_idx, []).append(mx_ip)

    # Efficient O(E) precomputed IP -> ASN mapping
    ip_to_asn_map: Dict[int, int] = dict(ip_belongs_asn)
    id_to_ip = {v: k for k, v in ip_to_id.items()}

    # Enrich nameserver metadata with glue IP -> ASN / geo / org details (1-hop promotion)
    ns_id_to_asns: Dict[int, set] = {}
    for ns_id, ip_id in ns_ns_ip:
        asn_id = ip_to_asn_map.get(ip_id)
        if asn_id is not None:
            ns_id_to_asns.setdefault(ns_id, set()).add(asn_id)
    for ns_host, meta_ns in ns_meta.items():
        for g_ip in meta_ns["glue_ips"]:
            g_ip_id = ip_to_id.get(g_ip)
            if g_ip_id is not None and g_ip_id in ip_to_asn_map:
                meta_ns["glue_asns"].add(ip_to_asn_map[g_ip_id])
            g_ip_meta = ip_meta.get(g_ip, {})
            cc = g_ip_meta.get("country_code") or g_ip_meta.get("country")
            if cc:
                meta_ns["glue_countries"].add(cc)
            org_s = g_ip_meta.get("org") or g_ip_meta.get("isp")
            if org_s:
                meta_ns["glue_orgs"].add(org_s)

    # Build Hop-1 virtual shortcut edges (primary_nt, uses_ns_asn, asn) strictly for no_ip domains
    d_uses_ns_asn: List[Tuple[int, int]] = []
    zone_to_ns_ids: Dict[int, List[int]] = {}
    for z_idx, ns_id in d_uses_ns:
        zone_to_ns_ids.setdefault(z_idx, []).append(ns_id)

    for d_idx in range(n_domains):
        if domain_has_resolved_ip[d_idx]:
            continue
        z_idx = fqdn_subdomain_apex[d_idx][1] if hierarchical_subdomains else d_idx
        if ns_id_to_asns:
            for ns_id in zone_to_ns_ids.get(z_idx, ()):
                for asn_id in ns_id_to_asns.get(ns_id, ()):
                    d_uses_ns_asn.append((d_idx, asn_id))
        # Bridge dormant MX IPs -> ASN
        for m_ip in domain_mx_ips.get(d_idx, ()):
            m_ip_id = ip_to_id.get(m_ip)
            if m_ip_id is not None and m_ip_id in ip_to_asn_map:
                d_uses_ns_asn.append((d_idx, ip_to_asn_map[m_ip_id]))
        # Bridge dormant CNAME target if present in graph as another domain or nameserver
        c_tgts = col_cname_targets[d_idx]
        if c_tgts is not None and isinstance(c_tgts, (list, np.ndarray)):
            for c_tgt in c_tgts:
                if c_tgt in ns_to_id:
                    for asn_id in ns_id_to_asns.get(ns_to_id[c_tgt], ()):
                        d_uses_ns_asn.append((d_idx, asn_id))
                elif c_tgt in domain_to_id:
                    tgt_d_id = domain_to_id[c_tgt]
                    tgt_ips = col_resolved_ips[tgt_d_id]
                    if isinstance(tgt_ips, (list, np.ndarray)):
                        for tip in tgt_ips:
                            tip_id = ip_to_id.get(tip)
                            if tip_id is not None and tip_id in ip_to_asn_map:
                                d_uses_ns_asn.append((d_idx, ip_to_asn_map[tip_id]))

    # Deduplicate edge list helper: returns contiguous [2, E] int64 tensor
    def to_edge_tensor(pairs: Any) -> torch.Tensor:
        if pairs is None or len(pairs) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        if isinstance(pairs, np.ndarray):
            if pairs.size == 0:
                return torch.empty((2, 0), dtype=torch.long)
            unique_pairs = np.unique(pairs, axis=0)
            return torch.from_numpy(unique_pairs.T).contiguous().to(torch.long)
        unique_pairs = sorted(set(pairs))
        arr = np.array(unique_pairs, dtype=np.int64).T
        return torch.from_numpy(arr).contiguous()

    # 5. Populate PyTorch Geometric HeteroData instance with multi-modal intrinsic node features
    base = HeteroData()
    base[primary_nt].raw_x = torch.tensor(raw_domain_feats, dtype=torch.float)
    base[primary_nt].char_bytes = domain_char_bytes
    base[primary_nt].y = torch.tensor(valid_df["label"].to_numpy(), dtype=torch.long)
    base[primary_nt].num_nodes = n_domains

    if hierarchical_subdomains:
        base["apex_domain"].raw_x = torch.from_numpy(build_apex_intrinsic_features(apex_to_id, apex_meta))
        base["apex_domain"].num_nodes = len(apex_to_id)

    base["ip"].raw_x = torch.from_numpy(build_ip_intrinsic_features(ip_to_id, ip_meta))
    base["ip"].num_nodes = len(ip_to_id)

    base["nameserver"].raw_x = torch.from_numpy(build_nameserver_intrinsic_features(ns_to_id, ns_meta))
    base["nameserver"].num_nodes = len(ns_to_id)

    base["registrar"].raw_x = torch.from_numpy(build_registrar_intrinsic_features(reg_to_id, reg_meta))
    base["registrar"].num_nodes = len(reg_to_id)

    base["asn"].raw_x = torch.from_numpy(build_asn_intrinsic_features(asn_to_id, asn_meta))
    base["asn"].num_nodes = len(asn_to_id)

    base["certificate"].raw_x = torch.from_numpy(build_certificate_intrinsic_features(cert_to_id, cert_meta))
    base["certificate"].num_nodes = len(cert_to_id)

    # 6. Assign forward and reverse relational edges (maintains bidirectional connectivity)
    if hierarchical_subdomains:
        e_sub = to_edge_tensor(fqdn_subdomain_apex)
        base["fqdn", "subdomain_of", "apex_domain"].edge_index = e_sub
        base["apex_domain", "rev_subdomain_of", "fqdn"].edge_index = torch.stack([e_sub[1], e_sub[0]]) if e_sub.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_res = to_edge_tensor(d_resolves_ip)
    base[primary_nt, "resolves_to", "ip"].edge_index = e_res
    base["ip", "rev_resolves_to", primary_nt].edge_index = torch.stack([e_res[1], e_res[0]]) if e_res.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_ns_ip = to_edge_tensor(ns_ns_ip)
    base["nameserver", "ns_ip", "ip"].edge_index = e_ns_ip
    base["ip", "rev_ns_ip", "nameserver"].edge_index = torch.stack([e_ns_ip[1], e_ns_ip[0]]) if e_ns_ip.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    ns_reg_src = "apex_domain" if hierarchical_subdomains else "domain"
    e_ns = to_edge_tensor(d_uses_ns)
    base[ns_reg_src, "uses_ns", "nameserver"].edge_index = e_ns
    base["nameserver", "rev_uses_ns", ns_reg_src].edge_index = torch.stack([e_ns[1], e_ns[0]]) if e_ns.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_reg = to_edge_tensor(d_registered_reg)
    base[ns_reg_src, "registered_by", "registrar"].edge_index = e_reg
    base["registrar", "rev_registered_by", ns_reg_src].edge_index = torch.stack([e_reg[1], e_reg[0]]) if e_reg.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_asn = to_edge_tensor(ip_belongs_asn)
    base["ip", "belongs_to_asn", "asn"].edge_index = e_asn
    base["asn", "rev_belongs_to_asn", "ip"].edge_index = torch.stack([e_asn[1], e_asn[0]]) if e_asn.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    e_cert = to_edge_tensor(d_uses_cert)
    base[primary_nt, "uses_cert", "certificate"].edge_index = e_cert
    base["certificate", "rev_uses_cert", primary_nt].edge_index = torch.stack([e_cert[1], e_cert[0]]) if e_cert.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    # Hop-1 shortcut edge for no_ip cold-start recovery
    e_ns_asn = to_edge_tensor(d_uses_ns_asn)
    base[primary_nt, "uses_ns_asn", "asn"].edge_index = e_ns_asn
    base["asn", "rev_uses_ns_asn", primary_nt].edge_index = torch.stack([e_ns_asn[1], e_ns_asn[0]]) if e_ns_asn.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    if include_bgp_prefix and len(prefix_to_id) > 0:
        base["prefix"].raw_x = torch.from_numpy(build_prefix_intrinsic_features(prefix_to_id))
        base["prefix"].num_nodes = len(prefix_to_id)
        e_pre = to_edge_tensor(ip_in_prefix)
        base["ip", "in_prefix", "prefix"].edge_index = e_pre
        base["prefix", "rev_in_prefix", "ip"].edge_index = torch.stack([e_pre[1], e_pre[0]]) if e_pre.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
        e_pasn = to_edge_tensor(prefix_belongs_asn)
        base["prefix", "belongs_to_asn", "asn"].edge_index = e_pasn
        base["asn", "rev_belongs_to_asn", "prefix"].edge_index = torch.stack([e_pasn[1], e_pasn[0]]) if e_pasn.numel() > 0 else torch.empty((2, 0), dtype=torch.long)

    # Map domain-indexed incident pairs for split-safe visible degree computation across all infra types
    if hierarchical_subdomains:
        apex_ns_map: Dict[int, List[int]] = {}
        for a_i, ns_i in d_uses_ns:
            apex_ns_map.setdefault(a_i, []).append(ns_i)
        apex_reg_map: Dict[int, List[int]] = {}
        for a_i, reg_i in d_registered_reg:
            apex_reg_map.setdefault(a_i, []).append(reg_i)
        dom_uses_ns_pairs = [(f_i, ns_i) for (f_i, a_i) in fqdn_subdomain_apex for ns_i in apex_ns_map.get(a_i, ())]
        dom_reg_pairs = [(f_i, r_i) for (f_i, a_i) in fqdn_subdomain_apex for r_i in apex_reg_map.get(a_i, ())]
    else:
        dom_uses_ns_pairs = d_uses_ns
        dom_reg_pairs = d_registered_reg

    d_asn_pairs = [(d, ip_to_asn_map[ip]) for (d, ip) in d_resolves_ip if ip in ip_to_asn_map]
    if d_uses_ns_asn:
        d_asn_pairs.extend(d_uses_ns_asn)

    # 7. Store raw domain pairs (primary_dom_id, infra_node_id) for fast downstream split degree standardizations
    base.incident_pairs = {
        "ip": d_resolves_ip,
        "nameserver": dom_uses_ns_pairs,
        "registrar": dom_reg_pairs,
        "asn": d_asn_pairs,
        "certificate": d_uses_cert,
    }
    if hierarchical_subdomains:
        base.incident_pairs["apex_domain"] = fqdn_subdomain_apex

    base.id_maps = {
        primary_nt: domain_to_id,
        "ip": ip_to_id,
        "nameserver": ns_to_id,
        "registrar": reg_to_id,
        "asn": {str(k): v for k, v in asn_to_id.items()},
        "certificate": cert_to_id,
        "prefix": prefix_to_id,
    }
    if hierarchical_subdomains:
        base.id_maps["apex_domain"] = apex_to_id
    base.primary_node_type = primary_nt
    base.cert_key_mode = cert_key_mode
    base.include_has_subdomain = include_sub
    base.feature_schema = {
        primary_nt: get_domain_feature_names(use_lexical=True, include_has_subdomain=include_sub),
        "infra": INFRA_FEATURE_SCHEMAS,
    }
    del id_to_ip

    # Log cert degree distribution
    if len(cert_to_id) > 0 and len(d_uses_cert) > 0:
        c_counts = Counter(p[1] for p in d_uses_cert)
        if c_counts:
            vals = list(c_counts.values())
            p50 = float(np.percentile(vals, 50))
            p90 = float(np.percentile(vals, 90))
            p99 = float(np.percentile(vals, 99))
            logger.info(f"Cert degree distribution ({cert_key_mode}): p50={p50:.1f}, p90={p90:.1f}, p99={p99:.1f}, max={max(vals)}")

    return base


def apply_split_scaling(
    base: HeteroData,
    train_indices: np.ndarray,
    val_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    degree_mode: str = "transductive",
    include_certificates: bool = True,
    drop_edge_type: Optional[str] = None,
    use_lexical: bool = True,
    mask_future_domains: bool = False,
    mask_disjoint_groups: bool = False,
    max_hub_fanout: Optional[int] = None,
) -> Tuple[HeteroData, Dict[str, Any]]:
    """
    Applies per-split z-score scaling fit strictly on training domains/FQDNs and train-incident infra nodes.
    Supports both flat `'domain'` graphs and RFC-001 hierarchical `'fqdn' -> 'apex_domain'` graphs,
    bounded mega-hub reverse fan-out capping (`max_hub_fanout`), and directional test isolation.
    """
    data = HeteroData()
    scaler_stats: Dict[str, Any] = {}
    primary_nt = get_primary_domain_node_type(base)

    # Step 1: Primary domain/FQDN feature matrix standardization (strictly train-fit)
    raw_domain_x = base[primary_nt].raw_x.numpy()
    if not use_lexical:
        nonlex_idx = get_feature_indices(
            primary_nt,
            group="non_lexical",
            include_has_subdomain=bool(getattr(base, "include_has_subdomain", False)),
            dim=raw_domain_x.shape[1],
        )
        raw_domain_x = raw_domain_x[:, nonlex_idx]

    domain_scaler = TrainOnlyStandardizer().fit(raw_domain_x, train_indices)
    scaled_domain_x = domain_scaler.transform(raw_domain_x)
    data[primary_nt].x = torch.tensor(scaled_domain_x, dtype=torch.float)
    data[primary_nt].y = base[primary_nt].y.clone()
    data[primary_nt].num_nodes = base[primary_nt].num_nodes
    if hasattr(base[primary_nt], "char_bytes") and base[primary_nt].char_bytes is not None:
        data[primary_nt].char_bytes = base[primary_nt].char_bytes.clone()
    scaler_stats[primary_nt] = domain_scaler.to_dict()
    if primary_nt != "domain":
        scaler_stats["domain"] = scaler_stats[primary_nt]

    # Step 2: Infrastructure (and apex_domain) node degree + IIF calculation and multi-feature standardization
    infra_types = []
    if "apex_domain" in base.node_types:
        infra_types.append("apex_domain")
    infra_types.extend(["ip", "nameserver", "registrar", "asn"])
    if include_certificates and "certificate" in base.node_types and base["certificate"].num_nodes > 0:
        infra_types.append("certificate")
    if "prefix" in base.node_types and base["prefix"].num_nodes > 0:
        infra_types.append("prefix")

    incident_pairs = getattr(base, "incident_pairs", {})

    # Construct boolean masks for O(1) vectorized indexing over domain/FQDN endpoints
    is_train_domain = np.zeros(base[primary_nt].num_nodes, dtype=bool)
    is_train_domain[train_indices] = True
    if val_indices is not None and len(val_indices) > 0:
        is_visible_domain = is_train_domain.copy()
        is_visible_domain[val_indices] = True
    else:
        is_visible_domain = is_train_domain

    n_vis_dom = max(1, int(np.sum(is_visible_domain)))
    log_denom = float(np.log1p(n_vis_dom))
    iif_by_type: Dict[str, torch.Tensor] = {}

    for nt in infra_types:
        n_nodes = base[nt].num_nodes
        has_raw_infra = hasattr(base[nt], "raw_x") and base[nt].raw_x is not None
        raw_infra_intrinsic = base[nt].raw_x.numpy() if has_raw_infra else None
        expected_dim = (raw_infra_intrinsic.shape[1] + 2) if has_raw_infra else INFRA_TOTAL_DIMS.get(nt, 2)

        if n_nodes == 0:
            data[nt].x = torch.empty((0, expected_dim), dtype=torch.float)
            data[nt].iif_weight = torch.empty((0,), dtype=torch.float)
            data[nt].num_nodes = 0
            iif_by_type[nt] = data[nt].iif_weight
            continue

        pairs_raw = incident_pairs.get(nt, [])
        if isinstance(pairs_raw, np.ndarray):
            pairs_arr = pairs_raw
        elif len(pairs_raw) > 0:
            pairs_arr = np.asarray(pairs_raw, dtype=np.int64).reshape(-1, 2)
        else:
            pairs_arr = np.empty((0, 2), dtype=np.int64)

        if len(pairs_arr) == 0:
            deg_counts = np.zeros(n_nodes, dtype=np.float32)
            vis_deg_counts = np.zeros(n_nodes, dtype=np.float32)
            train_incident = np.arange(n_nodes)
        else:
            mask_vis = is_visible_domain[pairs_arr[:, 0]]
            vis_deg_counts = np.bincount(pairs_arr[mask_vis, 1], minlength=n_nodes).astype(np.float32)
            if degree_mode == "train_visible":
                deg_counts = vis_deg_counts
            else:
                deg_counts = np.bincount(pairs_arr[:, 1], minlength=n_nodes).astype(np.float32)

            train_mask = is_train_domain[pairs_arr[:, 0]]
            train_incident = np.unique(pairs_arr[train_mask, 1])
            if len(train_incident) == 0:
                train_incident = np.arange(n_nodes)

        log_degs = np.log1p(deg_counts)[:, None]
        iif_vals = (np.log1p(n_vis_dom / np.maximum(1.0, vis_deg_counts)) / max(log_denom, 1e-6)).astype(np.float32)
        iif_col = iif_vals[:, None]

        if has_raw_infra and raw_infra_intrinsic.shape[0] == n_nodes:
            combined_raw = np.hstack([log_degs, iif_col, raw_infra_intrinsic]).astype(np.float32)
        else:
            combined_raw = np.hstack([log_degs, iif_col]).astype(np.float32)

        infra_scaler = TrainOnlyStandardizer().fit(combined_raw, train_incident)
        scaled_infra = infra_scaler.transform(combined_raw)
        data[nt].x = torch.tensor(scaled_infra, dtype=torch.float)
        data[nt].iif_weight = torch.tensor(iif_vals, dtype=torch.float)
        data[nt].num_nodes = n_nodes
        iif_by_type[nt] = data[nt].iif_weight
        scaler_stats[nt] = infra_scaler.to_dict()

    # Step 3: Relational edge assembly with topology isolation, bounded fan-out & IIF edge weights
    test_set = set(test_indices.tolist()) if test_indices is not None else set()
    is_domain_test: Optional[torch.Tensor] = None
    if len(test_set) > 0 and (mask_future_domains or mask_disjoint_groups):
        is_domain_test = torch.zeros(base[primary_nt].num_nodes, dtype=torch.bool)
        is_domain_test[torch.tensor(list(test_set), dtype=torch.long)] = True

    for edge_type in base.edge_types:
        src_t, rel, dst_t = edge_type
        if not include_certificates and ("certificate" in (src_t, dst_t)):
            continue
        if drop_edge_type:
            if drop_edge_type in rel or rel == drop_edge_type:
                continue
            if drop_edge_type in ("belongs_to_asn", "uses_ns") and "uses_ns_asn" in rel:
                continue

        edge_tensor = base[edge_type].edge_index.clone()

        if is_domain_test is not None:
            if is_domain_test.device != edge_tensor.device:
                is_domain_test = is_domain_test.to(edge_tensor.device)
            # RFC-001 Section 4 Invariant 3: keep ("fqdn", "subdomain_of", "apex_domain") active
            # while directionally isolating test primary nodes from shared infrastructure hubs.
            is_subdomain_hierarchy_edge = (rel in ("subdomain_of", "rev_subdomain_of"))
            if not (primary_nt == "fqdn" and is_subdomain_hierarchy_edge):
                if src_t == primary_nt:
                    mask = ~is_domain_test[edge_tensor[0]]
                    edge_tensor = edge_tensor[:, mask]
                elif dst_t == primary_nt:
                    mask = ~is_domain_test[edge_tensor[1]]
                    edge_tensor = edge_tensor[:, mask]

        # RFC-003 Section 3.3: Bounded neighbor fan-out cap on mega-hubs (deg > max_hub_fanout)
        if max_hub_fanout is not None and max_hub_fanout > 0 and edge_tensor.shape[1] > max_hub_fanout:
            if rel.startswith("rev_") and src_t not in ("domain", "fqdn"):
                src_np = edge_tensor[0].cpu().numpy()
                counts = np.bincount(src_np, minlength=base[src_t].num_nodes)
                if np.any(counts > max_hub_fanout):
                    seen_cnt: Dict[int, int] = {}
                    keep_mask = np.ones(len(src_np), dtype=bool)
                    for e_pos, s_id in enumerate(src_np):
                        c = seen_cnt.get(int(s_id), 0)
                        if c >= max_hub_fanout:
                            keep_mask[e_pos] = False
                        else:
                            seen_cnt[int(s_id)] = c + 1
                    edge_tensor = edge_tensor[:, torch.from_numpy(keep_mask).to(edge_tensor.device)]

        data[edge_type].edge_index = edge_tensor

        # Attach continuous IIF edge weights aligned with edge_tensor
        if edge_tensor.shape[1] == 0:
            data[edge_type].edge_weight = torch.empty((0,), dtype=torch.float)
        elif src_t == primary_nt and dst_t in iif_by_type:
            data[edge_type].edge_weight = iif_by_type[dst_t][edge_tensor[1]].clone()
        elif dst_t == primary_nt and src_t in iif_by_type:
            data[edge_type].edge_weight = iif_by_type[src_t][edge_tensor[0]].clone()
        elif dst_t in iif_by_type:
            data[edge_type].edge_weight = iif_by_type[dst_t][edge_tensor[1]].clone()
        elif src_t in iif_by_type:
            data[edge_type].edge_weight = iif_by_type[src_t][edge_tensor[0]].clone()

    data.id_maps = getattr(base, "id_maps", {})
    data.primary_node_type = primary_nt
    return data, scaler_stats


def build_hetero_graph(
    df: pd.DataFrame,
    train_indices: Optional[np.ndarray] = None,
    t_max: Optional[str] = None,
    use_lexical: bool = True,
    include_has_subdomain: bool = False,
    include_certificates: bool = True,
    hierarchical_subdomains: bool = False,
) -> Tuple[HeteroData, TrainOnlyStandardizer, Dict[str, Dict[str, int]]]:
    """
    Backwards-compatible wrapper.
    Builds base graph and applies split scaling in one step.
    """
    cfg = {
        "graph": {"cert_key_mode": "cn", "hierarchical_subdomains": hierarchical_subdomains},
        "parsing": {"include_has_subdomain": include_has_subdomain, "preserve_subdomains": hierarchical_subdomains},
    }
    base = build_base_graph(df, config=cfg, t_max=t_max)
    primary_nt = get_primary_domain_node_type(base)
    fit_idx = train_indices if train_indices is not None else np.arange(base[primary_nt].num_nodes)
    scaled, stats = apply_split_scaling(
        base=base,
        train_indices=fit_idx,
        include_certificates=include_certificates,
        use_lexical=use_lexical
    )

    scaler = TrainOnlyStandardizer()
    scaler.mean = np.array(stats[primary_nt]["mean"])
    scaler.std = np.array(stats[primary_nt]["std"])

    return scaled, scaler, base.id_maps


def save_graph_cohort_bundle(
    graph: HeteroData,
    df: pd.DataFrame,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Saves an atomic graph-cohort bundle:
    1. heterodata.pt (base unscaled graph)
    2. domains_cohort.parquet (exact aligned domain dataframe, row index == domain node ID)
    3. id_maps.json (node name to ID mappings)
    4. graph_manifest.json (cryptographic checksums and counts to prevent stale/misaligned downstream runs)
    """
    import hashlib
    from src.hdg.cache import atomic_save_torch, atomic_save_json, compute_dict_fingerprint

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_domains = len(df)
    primary_nt = get_primary_domain_node_type(graph)
    if graph[primary_nt].num_nodes != n_domains:
        raise ValueError(
            f"Cannot save aligned graph bundle: graph {primary_nt} count ({graph[primary_nt].num_nodes}) "
            f"does not match dataframe row count ({n_domains})."
        )

    # 1. Compute domain identity and label checksums
    domain_names = df["domain"].astype(str).tolist()
    names_str = "\n".join(domain_names).encode("utf-8")
    dom_sha256 = hashlib.sha256(names_str).hexdigest()

    labels_arr = df["label"].to_numpy().astype(np.int64)
    labels_sha256 = hashlib.sha256(labels_arr.tobytes()).hexdigest()

    node_counts = {nt: int(graph[nt].num_nodes) for nt in graph.node_types}
    edge_counts = {f"{et[0]}__{et[1]}__{et[2]}": int(graph[et].edge_index.shape[1]) for et in graph.edge_types}

    feature_dims_dict: Dict[str, int] = {}
    for nt in graph.node_types:
        if nt in ("domain", "fqdn"):
            feature_dims_dict[nt] = int(graph[nt].raw_x.shape[1]) if hasattr(graph[nt], "raw_x") and graph[nt].raw_x is not None else int(graph[nt].x.shape[1])
        elif hasattr(graph[nt], "raw_x") and graph[nt].raw_x is not None:
            feature_dims_dict[nt] = int(graph[nt].raw_x.shape[1] + 2)
        elif hasattr(graph[nt], "x") and graph[nt].x is not None:
            feature_dims_dict[nt] = int(graph[nt].x.shape[1])
        else:
            feature_dims_dict[nt] = INFRA_TOTAL_DIMS.get(nt, 2)

    manifest = {
        "schema_version": "3.0.0",
        "primary_node_type": primary_nt,
        "domain_count": n_domains,
        "domain_names_sha256": dom_sha256,
        "labels_sha256": labels_sha256,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "feature_dim": feature_dims_dict.get(primary_nt, 30),
        "feature_dims_dict": feature_dims_dict,
        "config_hash": compute_dict_fingerprint(config or {}),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "extra_metadata": metadata or {}
    }
    manifest["manifest_id"] = compute_dict_fingerprint(manifest)

    # 2. Atomic serialization
    graph_pt_path = output_dir / "heterodata.pt"
    cohort_parquet_path = output_dir / "domains_cohort.parquet"
    manifest_json_path = output_dir / "graph_manifest.json"
    id_maps_path = output_dir / "id_maps.json"

    atomic_save_torch(graph, graph_pt_path)
    df.to_parquet(cohort_parquet_path, index=False)
    atomic_save_json(manifest, manifest_json_path)

    if hasattr(graph, "id_maps") and graph.id_maps:
        atomic_save_json(graph.id_maps, id_maps_path)

    logger.info(
        f"Saved aligned graph bundle to {output_dir} (domains={n_domains}, manifest_id={manifest['manifest_id']})"
    )
    return manifest


def load_aligned_graph_cohort(
    processed_dir: Path,
    require_manifest: bool = False
) -> Tuple[HeteroData, pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Loads and strictly validates graph-domain cohort alignment.
    Guarantees that dataframe rows, labels, and graph node indices match 1-to-1.
    """
    processed_dir = Path(processed_dir)
    graph_path = processed_dir / "heterodata.pt"

    # Prefer domains_cohort.parquet; fallback to domains.parquet
    cohort_path = processed_dir / "domains_cohort.parquet"
    if cohort_path.exists():
        df = pd.read_parquet(cohort_path)
    else:
        dom_path = processed_dir / "domains.parquet"
        if not dom_path.exists():
            raise FileNotFoundError(
                f"No domain dataset found in {processed_dir}. Expected 'domains_cohort.parquet' or 'domains.parquet'."
            )
        df = pd.read_parquet(dom_path)

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph artifact not found: {graph_path}")

    graph = torch.load(graph_path, weights_only=False)
    primary_nt = get_primary_domain_node_type(graph)

    # 1. Structural node count check
    if graph[primary_nt].num_nodes != len(df):
        raise ValueError(
            f"FATAL ALIGNMENT MISMATCH: graph['{primary_nt}'].num_nodes ({graph[primary_nt].num_nodes}) "
            f"!= len(df) ({len(df)}) in {processed_dir}. "
            "Graph was built from a different cohort than the loaded dataframe!"
        )

    # 2. Label agreement check
    y_graph = graph[primary_nt].y.cpu().numpy().astype(int)
    y_df = df["label"].to_numpy().astype(int)
    if not np.array_equal(y_graph, y_df):
        mismatches = int(np.sum(y_graph != y_df))
        raise ValueError(
            f"FATAL LABEL MISMATCH: {mismatches} of {len(df)} domain labels disagree between "
            f"heterodata.pt and dataframe in {processed_dir}."
        )

    # 3. Manifest checksum validation
    manifest_path = processed_dir / "graph_manifest.json"
    manifest: Optional[Dict[str, Any]] = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        import hashlib
        names_str = "\n".join(df["domain"].astype(str).tolist()).encode("utf-8")
        actual_dom_sha = hashlib.sha256(names_str).hexdigest()
        expected_dom_sha = manifest.get("domain_names_sha256")

        if expected_dom_sha and actual_dom_sha != expected_dom_sha:
            raise ValueError(
                f"FATAL COHORT CORRUPTION: Domain names SHA-256 ({actual_dom_sha}) does not match "
                f"manifest ({expected_dom_sha}) in {processed_dir}!"
            )
    elif require_manifest:
        raise FileNotFoundError(f"Missing mandatory graph manifest: {manifest_path}")

    # Ensure unscaled base graph has x populated if only raw_x exists
    if not hasattr(graph[primary_nt], "x") or graph[primary_nt].x is None:
        if hasattr(graph[primary_nt], "raw_x") and graph[primary_nt].raw_x is not None:
            graph[primary_nt].x = graph[primary_nt].raw_x.clone()

    for nt in ["apex_domain", "ip", "nameserver", "registrar", "asn", "certificate", "prefix"]:
        if nt in graph.node_types:
            if not hasattr(graph[nt], "x") or graph[nt].x is None:
                if hasattr(graph[nt], "raw_x") and graph[nt].raw_x is not None:
                    zeros_split = torch.zeros((graph[nt].num_nodes, 2), dtype=torch.float)
                    graph[nt].x = torch.cat([zeros_split, graph[nt].raw_x], dim=1)
                else:
                    dim = INFRA_TOTAL_DIMS.get(nt, 2)
                    graph[nt].x = torch.zeros((graph[nt].num_nodes, dim), dtype=torch.float)

    return graph, df, manifest


def build_temporal_graph_pair(
    df: pd.DataFrame,
    train_cutoff: Optional[str] = None,
    test_cutoff: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    month_col: str = "t_month",
    train_cutoff_month: Optional[str] = None,
    target_month: Optional[str] = None,
    temporal_col: Optional[str] = None
) -> Tuple[HeteroData, HeteroData, Dict[str, Any]]:
    """
    Constructs a leakage-safe temporal graph pair:
    - train_graph: Built strictly from records with t_month <= train_cutoff.
      Zero future domains, zero future edges, zero future infrastructure entities exist in train_graph.
    - eval_graph: Cumulative graph up to target_month / test_cutoff.
    Returns: (train_graph, eval_graph, meta_dict)
    """
    col = temporal_col or month_col
    t_cut = train_cutoff_month or train_cutoff
    e_cut = target_month or test_cutoff or t_cut

    if t_cut is None:
        raise ValueError("Must provide train_cutoff or train_cutoff_month.")

    if col not in df.columns:
        from src.hdg.data.temporal import extract_month_str
        df = df.copy()
        df[col] = df["t"].apply(extract_month_str)

    train_df = df[df[col] <= t_cut].copy().reset_index(drop=True)
    if len(train_df) == 0:
        raise ValueError(f"No records found with {col} <= '{t_cut}'.")

    eval_df = df[df[col] <= e_cut].copy().reset_index(drop=True)

    train_graph = build_base_graph(train_df, config=config)
    eval_graph = build_base_graph(eval_df, config=config)

    meta = {
        "train_cutoff_month": t_cut,
        "target_month": e_cut,
        "train_domain_count": len(train_df),
        "target_domain_count": len(eval_df) - len(train_df),
        "train_df": train_df,
        "eval_df": eval_df
    }

    return train_graph, eval_graph, meta

