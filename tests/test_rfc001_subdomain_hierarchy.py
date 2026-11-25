"""
Comprehensive unit & integration tests for RFC-001 (Subdomain Preservation & Dual-Tier Hierarchy)
and RFC-003 polish (dormant MX bridging + bounded mega-hub reverse fan-out):
- Gate 1: preserve_subdomains retains sibling FQDNs + exports domains_fqdn.parquet & domains_apex.parquet
- Gate 2: stratified_apex_cluster_split guarantees strict zero-apex overlap & within_apex_sibling_split works
- Gate 3: Hierarchical HeteroData (`fqdn` + `apex_domain` D=16 + `subdomain_of` edges + directional isolation + max_hub_fanout)
- Gate 4: HeteroGNN + build_ego_subgraph + train_eval_gnn end-to-end execution on hierarchical graphs
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.hdg.data.features import INFRA_TOTAL_DIMS, get_primary_domain_node_type
from src.hdg.data.graph import (
    apply_split_scaling,
    build_base_graph,
    load_aligned_graph_cohort,
    save_graph_cohort_bundle,
)
from src.hdg.data.parse import parse_and_process_dataset
from src.hdg.data.splits import stratified_apex_cluster_split, within_apex_sibling_split
from src.hdg.eval.latency import build_ego_subgraph
from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.training.train import train_eval_gnn


def test_preserve_subdomains_and_dual_parquet_export(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create raw JSON feeds where sibling FQDNs share identical apex domains
    def _rec(dom: str, src: str, month_day: str, mal: str = None, no_ip: bool = False):
        return {
            "domain_name": dom,
            "source": src,
            "malware_type": mal,
            "sourced_on": {"$date": f"2023-08-{month_day}T00:00:00Z"},
            "evaluated_on": {"$date": f"2023-08-{month_day}T00:00:00Z"},
            "dns": {
                "A": None if no_ip else ["198.51.100.10"],
                "NS": {"ns1.shared.net": {"related_ips": [{"value": "198.51.100.20"}]}},
                "MX": {"mx1.shared.net": {"related_ips": [{"value": "198.51.100.30"}]}} if no_ip else None,
            },
            "rdap": {"registration_date": {"$date": "2022-01-01T00:00:00Z"}, "entities": {"registrar": [{"name": "RegOne"}]}},
            "ip_data": [
                {"ip": "198.51.100.10", "from_record": "A", "asn": {"asn": 13335, "as_org": "Cloudflare", "prefix_len": 24}},
                {"ip": "198.51.100.30", "from_record": "MX", "asn": {"asn": 24940, "as_org": "Hetzner", "prefix_len": 24}},
            ],
        }

    mal_recs = [
        _rec("login.evil-corp.com", "ThreatFox", "01", mal="phishing"),
        _rec("auth.evil-corp.com", "ThreatFox", "02", mal="phishing"),
        _rec("c2.bad-actor.net", "URLhaus", "03", mal="cobalt", no_ip=True),
    ]
    umb_recs = [
        _rec("www.good-org.com", "Umbrella", "01"),
        _rec("mail.good-org.com", "Umbrella", "02"),
        _rec("docs.safe-cloud.org", "Umbrella", "03"),
    ]
    ces_recs = [
        _rec("portal.uni-example.edu", "CESNET", "01"),
        _rec("cdn.uni-example.edu", "CESNET", "02"),
    ]

    with open(raw_dir / "malware.json", "w", encoding="utf-8") as f:
        json.dump(mal_recs, f)
    with open(raw_dir / "benign_umbrella.json", "w", encoding="utf-8") as f:
        json.dump(umb_recs, f)
    with open(raw_dir / "benign_cesnet.json", "w", encoding="utf-8") as f:
        json.dump(ces_recs, f)

    cfg = {
        "randomness": {"seed": 42},
        "parsing": {
            "preserve_subdomains": True,
            "normalize_to_e2LD": False,
            "benign_ratio": None,
            "asn_candidate_keys": ["asn", "number"],
            "registrar_candidate_keys": ["name"],
        },
    }
    summary = parse_and_process_dataset(raw_dir, out_dir, config=cfg)

    # All 8 sibling FQDNs MUST be preserved instead of collapsing down to 5 apex domains!
    assert summary["preserve_subdomains"] is True
    assert summary["final_counts"]["total_primary"] == 8
    assert summary["final_counts"]["apex_count"] == 5
    assert (out_dir / "domains_fqdn.parquet").exists()
    assert (out_dir / "domains_apex.parquet").exists()

    df_fqdn = pd.read_parquet(out_dir / "domains_fqdn.parquet")
    df_apex = pd.read_parquet(out_dir / "domains_apex.parquet")
    assert set(["domain_id", "fqdn", "apex_id", "label", "f_length", "f_subdomain_count"]).issubset(df_fqdn.columns)
    assert set(["apex_id", "apex_domain", "tld", "subdomain_fanout_count", "is_public_suffix"]).issubset(df_apex.columns)
    assert set(df_fqdn["apex_id"].unique()).issubset(set(df_apex["apex_id"].unique()))


def test_apex_cluster_and_within_apex_splits_and_hierarchical_gnn(tmp_path: Path) -> None:
    rows = []
    # Build 12 apex clusters (6 benign, 6 malicious), 3 sibling FQDNs each = 36 FQDNs
    for a_idx in range(12):
        is_mal = (a_idx >= 6)
        apex_dom = f"cluster{a_idx}.{'ru' if is_mal else 'com'}"
        for s_idx in range(3):
            fqdn = f"sub{s_idx}.{apex_dom}"
            is_no_ip = (s_idx == 2)
            rows.append({
                "domain": fqdn,
                "raw_domain": fqdn,
                "apex_domain": apex_dom,
                "e2LD": apex_dom,
                "label": 1 if is_mal else 0,
                "source": "ThreatFox" if is_mal else "Umbrella",
                "t": "2023-08-10T00:00:00Z",
                "t_reg": "2023-07-01T00:00:00Z" if is_mal else "2018-01-01T00:00:00Z",
                "no_ip": is_no_ip,
                "no_rdap": False,
                "no_tls": False,
                "nameservers": ["ns1.hub.net" if a_idx < 8 else f"ns{a_idx}.ded.net"],
                "registrar": f"Reg{a_idx % 2}",
                "leaf_cert_key_cn": f"cert_{a_idx}_{s_idx}",
                "resolved_ips": [] if is_no_ip else [f"198.51.100.{a_idx % 4}"],
                "ip_records": [
                    {"ip": f"198.51.100.{a_idx % 4}", "asn": 13335 + (a_idx % 2), "asn_org": "Cloudflare", "country_code": "US", "prefix_len": 24},
                    {"ip": "198.51.100.99", "asn": 24940, "asn_org": "Hetzner", "country_code": "DE", "prefix_len": 24},
                ],
                "ns_ip_pairs": [("ns1.hub.net", f"198.51.100.{a_idx % 4}")],
                "mx_ip_pairs": [("mx1.dor.net", "198.51.100.99")] if is_no_ip else [],
            })
    df = pd.DataFrame(rows)

    # 1. Leak-free Apex-Cluster Split
    apex_splits = stratified_apex_cluster_split(df, train_ratio=0.66, val_ratio=0.17, test_ratio=0.17, seed=42)
    apex_splits.assert_disjoint()
    tr_ap = set(df.iloc[apex_splits.train_indices]["apex_domain"])
    va_ap = set(df.iloc[apex_splits.val_indices]["apex_domain"])
    te_ap = set(df.iloc[apex_splits.test_indices]["apex_domain"])
    assert len(tr_ap.intersection(te_ap)) == 0
    assert len(tr_ap.intersection(va_ap)) == 0
    assert len(va_ap.intersection(te_ap)) == 0

    # 2. Within-Apex Sibling Split
    sib_splits = within_apex_sibling_split(df, seed=42)
    sib_splits.assert_disjoint()

    # 3. Hierarchical HeteroData Construction & Scaling
    cfg = {"parsing": {"preserve_subdomains": True}, "graph": {"hierarchical_subdomains": True, "max_hub_fanout": 4}}
    base = build_base_graph(df, config=cfg)
    assert get_primary_domain_node_type(base) == "fqdn"
    assert "apex_domain" in base.node_types
    assert ("fqdn", "subdomain_of", "apex_domain") in base.edge_types
    assert ("apex_domain", "uses_ns", "nameserver") in base.edge_types
    assert ("apex_domain", "registered_by", "registrar") in base.edge_types
    assert base["apex_domain"].num_nodes == 12
    assert base["fqdn"].num_nodes == 36
    # Dormant MX bridging gave no_ip FQDNs uses_ns_asn shortcut edges
    assert base["fqdn", "uses_ns_asn", "asn"].edge_index.shape[1] > 0

    scaled, _ = apply_split_scaling(
        base,
        apex_splits.train_indices,
        val_indices=apex_splits.val_indices,
        test_indices=apex_splits.test_indices,
        mask_disjoint_groups=True,
        max_hub_fanout=4,
    )
    assert scaled["apex_domain"].x.shape[1] == INFRA_TOTAL_DIMS["apex_domain"] == 16
    # Directional isolation check (RFC-001 Section 4):
    # Test FQDNs retain subdomain_of -> apex_domain, while test resolves_to -> ip is isolated!
    e_sub_src = set(scaled["fqdn", "subdomain_of", "apex_domain"].edge_index[0].tolist())
    assert any(t in e_sub_src for t in apex_splits.test_indices)
    e_res_src = set(scaled["fqdn", "resolves_to", "ip"].edge_index[0].tolist())
    assert not any(t in e_res_src for t in apex_splits.test_indices)
    # Mega-hub reverse edge fan-out cap <= 4 per infra source node
    rev_res = scaled["ip", "rev_resolves_to", "fqdn"].edge_index
    if rev_res.numel() > 0:
        assert int(torch.bincount(rev_res[0]).max().item()) <= 4

    # Cohort bundle round-trip alignment check with fqdn primary node type
    save_graph_cohort_bundle(base, df, tmp_path, config=cfg)
    loaded_g, loaded_df, manifest = load_aligned_graph_cohort(tmp_path, require_manifest=True)
    assert manifest["primary_node_type"] == "fqdn"
    assert len(loaded_df) == loaded_g["fqdn"].num_nodes

    # 4. Hierarchical HeteroGNN + Ego-subgraph + Training Loop
    in_ch = {nt: scaled[nt].x.shape[1] for nt in scaled.node_types}
    model = HeteroGNN(
        metadata=scaled.metadata(),
        in_channels_dict=in_ch,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
        variant="sage",
        use_gated_agg=True,
        use_char_cnn=True,
        char_cnn_dim=8,
        use_completeness_gate=True,
    )
    assert model.target_node_type == "fqdn"
    model.eval()
    ew = {et: scaled[et].edge_weight for et in scaled.edge_types if hasattr(scaled[et], "edge_weight")}
    with torch.no_grad():
        full_out = model(scaled.x_dict, scaled.edge_index_dict, edge_weight_dict=ew, char_bytes=scaled["fqdn"].char_bytes)
        ego, loc = build_ego_subgraph(scaled, [0], num_hops=2)
        ew_e = {et: ego[et].edge_weight for et in ego.edge_types if hasattr(ego[et], "edge_weight")}
        sub_out = model(ego.x_dict, ego.edge_index_dict, edge_weight_dict=ew_e, char_bytes=ego["fqdn"].char_bytes)
    assert np.isclose(
        float(F.softmax(full_out[0:1], dim=-1)[0, 1].item()),
        float(F.softmax(sub_out[loc], dim=-1)[0, 1].item()),
        atol=1e-5,
    )

    test_eval, _, _, _ = train_eval_gnn(
        model=model,
        data=scaled,
        splits=apex_splits,
        df=df,
        epochs=3,
        patience=2,
        min_epochs=1,
        bootstrap_samples=10,
        loss_type="focal",
        threshold_criterion="f0.5",
        use_temperature_scaling=True,
        seed=42,
    )
    assert "roc_auc" in test_eval
