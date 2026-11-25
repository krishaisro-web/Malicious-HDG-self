"""
Comprehensive unit tests for RFC-003:
- Gate 1: Zenodo ASN/BGP key fix (as_org, network_address, prefix_len) and preserved metadata
- Gate 2: Named feature schema registry, TLD-stripped lexical features, multi-feature infra raw_x,
          IIF weighting, Hop-1 uses_ns_asn shortcut, uint8 char_bytes, dynamic empty node dims
- Gate 3: GatedSAGEConvLayer, CharCNNBranch, neighborhood-completeness gate, GRL, Deep CORAL,
          ego-subgraph equivalence with RFC-003 features
- Gate 4: Source drift audit, t_reg birth-cohort split, 1-hop infra features for XGBoost
"""

from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.hdg.data.audit import audit_source_drift
from src.hdg.data.features import (
    INFRA_TOTAL_DIMS,
    extract_domain_body,
    extract_domain_features_vectorized,
    get_domain_feature_names,
    get_feature_indices,
)
from src.hdg.data.graph import (
    apply_split_scaling,
    build_base_graph,
    save_graph_cohort_bundle,
)
from src.hdg.data.parse import resolve_asn, stream_parse_file
from src.hdg.data.splits import DataSplits, random_stratified_split
from src.hdg.data.temporal import registration_birth_cohort_split
from src.hdg.eval.latency import build_ego_subgraph
from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.training.baselines import extract_1hop_infra_features_for_tabular
from src.hdg.training.train import deep_coral_loss, train_eval_gnn
import tldextract


def test_resolve_asn_zenodo_keys_and_legacy_fallback() -> None:
    cnt = Counter()
    # Zenodo DomainRadar v2 keys
    asn_n, org, pfx = resolve_asn(
        {"asn": 13335, "as_org": "Cloudflare, Inc.", "network_address": "104.16.0.0", "prefix_len": 12},
        ["asn", "autonomous_system_number", "number"],
        cnt,
    )
    assert asn_n == 13335
    assert org == "Cloudflare, Inc."
    assert pfx == "104.16.0.0/12"

    # Legacy fallback
    asn_n2, org2, pfx2 = resolve_asn(
        {"number": 15169, "organization": "Google LLC", "network": "8.8.8.0/24"},
        ["asn", "autonomous_system_number", "number"],
        cnt,
    )
    assert asn_n2 == 15169
    assert org2 == "Google LLC"
    assert pfx2 == "8.8.8.0/24"


def test_stream_parser_preserves_zenodo_rfc003_fields() -> None:
    extractor = tldextract.TLDExtract(suffix_list_urls=())
    asn_c = Counter()
    reg_c = Counter()

    records = [
        {
            "domain_name": "portal.paypal-verify.fr",
            "source": "ThreatFox",
            "malware_type": "phishing",
            "sourced_on": {"$date": "2023-08-01T00:00:00Z"},
            "evaluated_on": {"$date": "2023-08-02T00:00:00Z"},
            "dns": {
                "A": None,
                "AAAA": None,
                "NS": {"ns1.bulletproof.net": {"related_ips": [{"value": "198.51.100.5"}]}},
                "SOA": {"primary_ns": "ns2.soa-only.net"},
                "CNAME": "alias.evil.net",
                "MX": {"mail.evil.net": {"priority": 10}},
                "TXT": ["v=spf1 ~all"],
                "ttls": {"A": 60, "NS": 3600},
                "remarks": {"has_spf": True, "has_dmarc": False, "has_dkim": False},
            },
            "rdap": {
                "registration_date": {"$date": "2023-07-20T00:00:00Z"},
                "entities": {
                    "registrar": [{"name": "ShadyReg", "whois_server": "whois.shady.net", "url": "https://shady.net", "tel": "+1"}],
                    "abuse": [{"email": "abuse@shady.net"}],
                },
            },
            "tls": {
                "protocol": "TLSv1.3",
                "cipher": "TLS_AES_256_GCM_SHA384",
                "certificates": [
                    {"common_name": "paypal-verify.fr", "organization": "Let's Encrypt", "country": "US", "valid_len": 90, "extension_count": 3, "is_root": False}
                ],
            },
            "ip_data": [
                {
                    "ip": "198.51.100.5",
                    "from_record": "NS",
                    "asn": {"asn": 24940, "as_org": "Hetzner Online GmbH", "network_address": "198.51.100.0", "prefix_len": 24},
                    "geo": {"country": "Germany", "country_code": "DE", "latitude": 50.1, "longitude": 8.6, "isp": "Hetzner", "org": "Hetzner"},
                }
            ],
        }
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(records, tmp)
        tmp_path = Path(tmp.name)

    try:
        parsed, _, _, _ = stream_parse_file(
            file_path=tmp_path,
            label=1,
            default_source="malware",
            extractor=extractor,
            normalize_to_e2ld=True,
            asn_candidate_keys=["asn", "autonomous_system_number", "number"],
            registrar_candidate_keys=["name", "handle", "organization"],
            asn_key_counter=asn_c,
            reg_key_counter=reg_c,
        )
        r = parsed[0]
        assert r["no_ip"] is True
        assert "ns2.soa-only.net" in r["nameservers"]
        assert r["cname_present"] is True
        assert r["soa_present"] is True
        assert r["reg_has_whois"] is True
        assert r["reg_has_email"] is True
        assert r["cert_org"] == "Let's Encrypt"
        assert r["cert_ext_count"] == 3
        assert r["ip_records"][0]["asn_org"] == "Hetzner Online GmbH"
        assert r["ip_records"][0]["country_code"] == "DE"
        assert r["ip_records"][0]["prefix_len"] == 24
        assert r["domain_body"] == "paypal-verify"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_tld_stripped_lexical_invariance_and_schema_registry() -> None:
    df = pd.DataFrame({
        "domain": ["paypal-verify.com", "paypal-verify.fr"],
        "label": [1, 1],
        "t": ["2023-08-01T00:00:00Z", "2023-08-01T00:00:00Z"],
        "t_reg": ["2023-07-01T00:00:00Z", "2023-07-01T00:00:00Z"],
    })
    assert extract_domain_body("paypal-verify.com") == extract_domain_body("paypal-verify.fr") == "paypal-verify"
    X = extract_domain_features_vectorized(df, use_lexical=True, include_has_subdomain=False)
    assert X.shape == (2, 30)
    assert len(get_domain_feature_names(use_lexical=True, include_has_subdomain=False)) == 30
    lex_idx = get_feature_indices("domain", group="lexical", dim=X.shape[1])
    assert len(lex_idx) == 8
    # Lexical features MUST be identical across .com vs .fr
    np.testing.assert_allclose(X[0, lex_idx], X[1, lex_idx], atol=1e-6)
    ctrl_idx = get_feature_indices("domain", group="controllable", dim=X.shape[1])
    assert len(ctrl_idx) > 0


def test_multimodal_graph_iif_ns_asn_and_empty_node_safety(tmp_path: Path) -> None:
    rows = []
    for i in range(16):
        is_no_ip = (i % 4 == 0)
        ns = "ns1.cloudflare.com" if i < 12 else f"ns{i}.dedicated.net"
        ip = "198.51.100.1" if not is_no_ip else None
        ns_ip = "198.51.100.99"
        rows.append({
            "domain": f"dom{i}.com",
            "domain_body": f"dom{i}",
            "label": i % 2,
            "source": "Umbrella" if i % 2 == 0 else "ThreatFox",
            "family": None if i % 2 == 0 else "emotet",
            "t": "2023-08-10T00:00:00Z",
            "t_eval": "2023-08-11T00:00:00Z",
            "t_reg": "2020-01-01T00:00:00Z" if i % 2 == 0 else "2023-08-01T00:00:00Z",
            "no_ip": is_no_ip,
            "no_rdap": False,
            "no_tls": True,
            "a_count": 0 if is_no_ip else 1,
            "aaaa_count": 0,
            "mx_count": 1,
            "has_spf": True,
            "has_dmarc": False,
            "has_dkim": False,
            "dnssec": False,
            "ttl_a": 300,
            "ttl_ns": 86400,
            "nameservers": [ns],
            "registrar": None,  # Intentionally empty registrar set to test dynamic 0-node dim!
            "tls_cert_count": 0,
            "leaf_cert_key_cn": None,
            "primary_asn": 13335,
            "asns": [13335],
            "resolved_ips": [] if is_no_ip else [ip],
            "ip_records": [
                {"ip": ns_ip, "from_record": "NS", "asn": 13335, "asn_org": "Cloudflare, Inc.", "country_code": "US", "prefix_len": 24, "prefix": "198.51.100.0/24"}
            ] + ([] if is_no_ip else [
                {"ip": ip, "from_record": "A", "asn": 13335, "asn_org": "Cloudflare, Inc.", "country_code": "US", "prefix_len": 24, "prefix": "198.51.100.0/24"}
            ]),
            "ns_ip_pairs": [(ns, ns_ip)],
        })
    df = pd.DataFrame(rows)
    base = build_base_graph(df)

    # Check uint8 char_bytes & Hop-1 uses_ns_asn for no_ip domains
    assert base["domain"].char_bytes.dtype == torch.uint8
    assert base["domain"].char_bytes.shape == (16, 64)
    assert ("domain", "uses_ns_asn", "asn") in base.edge_types
    assert base["domain", "uses_ns_asn", "asn"].edge_index.shape[1] > 0

    splits = random_stratified_split(df["label"].to_numpy(), seed=42)
    scaled, _ = apply_split_scaling(base, splits.train_indices, val_indices=splits.val_indices, test_indices=splits.test_indices)

    # Check dynamic multi-feature infra dimensions (including 0-node registrar!)
    for nt, exp_dim in INFRA_TOTAL_DIMS.items():
        if nt in scaled.node_types:
            assert scaled[nt].x.shape[1] == exp_dim

    # Mega-hub NS (deg >= 12) should receive strictly lower IIF weight than dedicated NS (deg=1)
    ns_iif = scaled["nameserver"].iif_weight.numpy()
    cf_id = base.id_maps["nameserver"]["ns1.cloudflare.com"]
    ded_id = base.id_maps["nameserver"]["ns12.dedicated.net"]
    assert ns_iif[cf_id] < ns_iif[ded_id]

    # Check manifest schema v3.0.0
    manifest = save_graph_cohort_bundle(base, df, tmp_path)
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["feature_dims_dict"]["ip"] == 12


def test_rfc003_model_extensions_grl_coral_and_ego_equivalence() -> None:
    rows = []
    for i in range(20):
        rows.append({
            "domain": f"site-{i}.com" if i % 2 == 0 else f"auth-x9k{i}.net",
            "label": i % 2,
            "source": "Umbrella" if i % 2 == 0 else "ThreatFox",
            "family": None if i % 2 == 0 else "emotet",
            "t": "2023-08-10T00:00:00Z",
            "t_reg": "2020-01-01T00:00:00Z" if i % 2 == 0 else "2023-08-01T00:00:00Z",
            "no_ip": (i % 5 == 0),
            "no_rdap": False,
            "no_tls": False,
            "nameservers": [f"ns{i % 3}.prov.com"],
            "registrar": f"Reg{i % 2}",
            "leaf_cert_key_cn": f"cert_{i % 2}",
            "cert_valid_len": 90,
            "resolved_ips": [] if (i % 5 == 0) else [f"198.51.100.{i % 4}"],
            "ip_records": [{"ip": f"198.51.100.{i % 4}", "asn": 13335 + (i % 2), "asn_org": "Cloudflare", "country_code": "US", "prefix_len": 24}],
            "ns_ip_pairs": [(f"ns{i % 3}.prov.com", f"198.51.100.{i % 4}")],
        })
    df = pd.DataFrame(rows)
    base = build_base_graph(df)
    splits = DataSplits(
        train_indices=np.arange(0, 12),
        val_indices=np.arange(12, 16),
        test_indices=np.arange(16, 20),
        split_type="test",
    )
    scaled, _ = apply_split_scaling(base, splits.train_indices, val_indices=splits.val_indices, test_indices=splits.test_indices)
    in_ch = {nt: scaled[nt].x.shape[1] for nt in scaled.node_types}

    torch.manual_seed(42)
    model = HeteroGNN(
        metadata=scaled.metadata(),
        in_channels_dict=in_ch,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
        variant="sage",
        use_gated_agg=True,
        use_char_cnn=True,
        char_cnn_dim=16,
        use_completeness_gate=True,
        num_envs=2,
    )
    model.eval()
    ew = {et: scaled[et].edge_weight for et in scaled.edge_types if hasattr(scaled[et], "edge_weight")}
    cb = scaled["domain"].char_bytes

    with torch.no_grad():
        full_logits = model(scaled.x_dict, scaled.edge_index_dict, edge_weight_dict=ew, char_bytes=cb)
        full_probs = F.softmax(full_logits, dim=-1)[:, 1].numpy()

    # Ego-subgraph equivalence must hold for GatedSAGE + IIF + Char-CNN + Completeness Gate!
    for d_idx in [0, 3, 5]:
        ego, loc = build_ego_subgraph(scaled, [d_idx], num_hops=2)
        ew_e = {et: ego[et].edge_weight for et in ego.edge_types if hasattr(ego[et], "edge_weight")}
        with torch.no_grad():
            sub_logits = model(ego.x_dict, ego.edge_index_dict, edge_weight_dict=ew_e, char_bytes=ego["domain"].char_bytes)
            sub_prob = float(F.softmax(sub_logits[loc], dim=-1)[0, 1].item())
        assert np.isclose(sub_prob, full_probs[d_idx], atol=1e-5)

    # Check Deep CORAL & full RFC-003 training loop
    c_loss = deep_coral_loss(torch.randn(32, 16), torch.randn(24, 16), max_samples=16)
    assert c_loss.item() >= 0.0

    eval_res, _, _, _ = train_eval_gnn(
        model=model,
        data=scaled,
        splits=splits,
        df=df,
        epochs=3,
        patience=2,
        min_epochs=1,
        bootstrap_samples=10,
        use_grl=True,
        use_coral=True,
        use_group_dro=True,
        drop_drift_shortcuts=True,
        co_primary_cross_source=True,
        seed=42,
    )
    assert "roc_auc" in eval_res


def test_drift_audit_treg_split_and_1hop_tabular(tmp_path: Path) -> None:
    rows = []
    for i in range(40):
        is_umb = (i % 2 == 0)
        rows.append({
            "domain": f"d{i}.com",
            "label": 0 if i < 24 else 1,
            "source": "Umbrella" if is_umb else ("CESNET" if i < 24 else "ThreatFox"),
            "t": "2023-08-01T00:00:00Z",
            "t_reg": "2020-05-01T00:00:00Z" if i < 24 else "2023-06-01T00:00:00Z",
            "no_ip": True if (is_umb and i < 24) else False,
            "no_rdap": False,
            "no_tls": False,
            "asns": [13335],
            "ip_records": [{"ip": "1.1.1.1", "asn_org": "Cloudflare, Inc.", "country_code": "US", "prefix_len": 24}],
            "nameservers": ["ns1.cloudflare.com"],
            "cert_org": "Let's Encrypt",
        })
    df = pd.DataFrame(rows)

    drift = audit_source_drift(df, output_dir=tmp_path, auc_threshold=0.70)
    assert (tmp_path / "drift_audit.json").exists()
    assert "no_ip" in drift["flagged_drift_features"]

    X_1hop = extract_1hop_infra_features_for_tabular(df)
    assert X_1hop.shape == (40, 10)

    # Mix labels in both <= 2022 and > 2022 for t_reg birth-cohort split
    df_treg = df.copy()
    df_treg.loc[0:9, "label"] = 1
    df_treg.loc[24:31, "label"] = 0
    treg_splits = registration_birth_cohort_split(df_treg, train_cutoff_year=2022, min_per_class=4, seed=42)
    treg_splits.assert_disjoint()
    assert len(treg_splits.train_indices) > 0 and len(treg_splits.test_indices) > 0
