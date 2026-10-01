"""
Self-contained pytest fixtures for Malicious-HDG test suite.
Provides a minimal synthetic graph with:
- 10 domains (5 benign, 5 malicious)
- 5 IPs
- 3 Nameservers
- 2 Registrars
- 2 ASNs
- 2 Certificates
Enables 100% of tests to pass on fresh checkouts without requiring data_raw/ or disk fixtures.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.fixture(scope="session")
def minimal_domains_df() -> pd.DataFrame:
    """Returns minimal DataFrame with 10 synthetic domains and complete metadata."""
    domains = [f"test-{i}.example.com" for i in range(10)]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    months = ["2023-05", "2023-05", "2023-05", "2023-05", "2023-06",
              "2023-06", "2023-06", "2023-07", "2023-07", "2023-07"]
    dates = [f"{m}-15T12:00:00Z" for m in months]
    asns = [13335, 13335, 13335, 15169, 15169, 15169, 13335, 15169, 13335, 15169]

    df = pd.DataFrame({
        "domain": domains,
        "raw_domain": domains,
        "e2LD": [d.split(".", 1)[-1] for d in domains],
        "label": labels,
        "source": ["Umbrella" if l == 0 else "ThreatFox" for l in labels],
        "family": [None if l == 0 else "emotet" for l in labels],
        "t": dates,
        "t_month": months,
        "no_ip": [False] * 10,
        "no_rdap": [False] * 10,
        "no_tls": [False] * 10,
        "primary_asn": asns,
        "nameservers": [["ns1.provider.com", "ns2.provider.com"]] * 10,
        "registrar": ["Registrar A" if i % 2 == 0 else "Registrar B" for i in range(10)],
        "leaf_cert_key": [f"cert_{i % 2}" for i in range(10)],
        "resolved_ips": [[f"198.51.100.{i % 5}"] for i in range(10)],
        "ip_records": [[{"ip": f"198.51.100.{i % 5}", "asn": asns[i]}] for i in range(10)],
        "ns_ip_pairs": [[("ns1.provider.com", "198.51.100.1")]] * 10
    })
    return df


@pytest.fixture(scope="session")
def minimal_synthetic_heterodata() -> HeteroData:
    """
    Constructs a minimal HeteroData graph satisfying:
    10 domains, 5 IPs, 3 Nameservers, 2 Registrars, 2 ASNs, 2 Certificates.
    """
    data = HeteroData()

    # Node attributes
    data["domain"].x = torch.randn((10, 17), dtype=torch.float)
    data["domain"].y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    data["domain"].num_nodes = 10

    data["ip"].x = torch.randn((5, 4), dtype=torch.float)
    data["ip"].num_nodes = 5

    data["nameserver"].x = torch.randn((3, 4), dtype=torch.float)
    data["nameserver"].num_nodes = 3

    data["registrar"].x = torch.randn((2, 4), dtype=torch.float)
    data["registrar"].num_nodes = 2

    data["asn"].x = torch.randn((2, 4), dtype=torch.float)
    data["asn"].num_nodes = 2

    data["certificate"].x = torch.randn((2, 4), dtype=torch.float)
    data["certificate"].num_nodes = 2

    # Edges: domain -> ip (resolves_to)
    d_ip_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    d_ip_dst = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    data["domain", "resolves_to", "ip"].edge_index = torch.tensor([d_ip_src, d_ip_dst], dtype=torch.long)
    data["ip", "rev_resolves_to", "domain"].edge_index = torch.tensor([d_ip_dst, d_ip_src], dtype=torch.long)

    # domain -> nameserver (uses_ns)
    d_ns_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    d_ns_dst = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    data["domain", "uses_ns", "nameserver"].edge_index = torch.tensor([d_ns_src, d_ns_dst], dtype=torch.long)
    data["nameserver", "rev_uses_ns", "domain"].edge_index = torch.tensor([d_ns_dst, d_ns_src], dtype=torch.long)

    # domain -> registrar (registered_by)
    d_reg_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    d_reg_dst = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    data["domain", "registered_by", "registrar"].edge_index = torch.tensor([d_reg_src, d_reg_dst], dtype=torch.long)
    data["registrar", "rev_registered_by", "domain"].edge_index = torch.tensor([d_reg_dst, d_reg_src], dtype=torch.long)

    # domain -> certificate (uses_cert)
    d_cert_src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    d_cert_dst = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    data["domain", "uses_cert", "certificate"].edge_index = torch.tensor([d_cert_src, d_cert_dst], dtype=torch.long)
    data["certificate", "rev_uses_cert", "domain"].edge_index = torch.tensor([d_cert_dst, d_cert_src], dtype=torch.long)

    # ip -> asn (belongs_to_asn)
    ip_asn_src = [0, 1, 2, 3, 4]
    ip_asn_dst = [0, 0, 0, 1, 1]
    data["ip", "belongs_to_asn", "asn"].edge_index = torch.tensor([ip_asn_src, ip_asn_dst], dtype=torch.long)
    data["asn", "rev_belongs_to_asn", "ip"].edge_index = torch.tensor([ip_asn_dst, ip_asn_src], dtype=torch.long)

    # nameserver -> ip (ns_ip)
    ns_ip_src = [0, 1, 2]
    ns_ip_dst = [0, 1, 2]
    data["nameserver", "ns_ip", "ip"].edge_index = torch.tensor([ns_ip_src, ns_ip_dst], dtype=torch.long)
    data["ip", "rev_ns_ip", "nameserver"].edge_index = torch.tensor([ns_ip_dst, ns_ip_src], dtype=torch.long)

    return data
