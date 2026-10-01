import torch
import pandas as pd
from torch_geometric.data import HeteroData
from pathlib import Path
from pandas.errors import EmptyDataError

def load_node_features(name, feature_cols, log_scale_cols=None):
    df = pd.read_csv(f"data_processed/graphs/nodes_{name}.csv").sort_values("node_id")
    log_scale_cols = log_scale_cols or []
    data = df[feature_cols].copy().astype(float)
    for col in log_scale_cols:
        data[col] = torch.log1p(torch.tensor(data[col].values, dtype=torch.float)).numpy()
    # z-score standardize every column (safe for binary 0/1 flags too — just low variance)
    for col in data.columns:
        std = data[col].std()
        mean = data[col].mean()
        if std > 1e-8:
            data[col] = (data[col] - mean) / std
        else:
            data[col] = data[col] - mean
    return torch.tensor(data.values, dtype=torch.float)

domain_x = load_node_features("domain", ["length", "num_labels", "tld_com", "tld_net", "tld_org", "tld_other", "nxdomain", "has_tls"])
ip_x = load_node_features("ip", ["ip_version", "fanin"], log_scale_cols=["fanin"])
ns_x = load_node_features("nameserver", ["fanin"], log_scale_cols=["fanin"])
reg_x = load_node_features("registrar", ["fanin"], log_scale_cols=["fanin"])
asn_x = load_node_features("asn", ["fanin"], log_scale_cols=["fanin"])

snapshot_dirs = sorted(
    [p for p in Path("data_processed/graphs").glob("snapshot_*") if p.is_dir()],
    key=lambda p: int(p.name.replace("snapshot_", ""))
)

for snap_dir in snapshot_dirs:
    data = HeteroData()
    data["domain"].x = domain_x
    data["ip"].x = ip_x
    data["nameserver"].x = ns_x
    data["registrar"].x = reg_x
    data["asn"].x = asn_x

    def edge_index(fname):
        try:
            df = pd.read_csv(snap_dir / fname)
        except EmptyDataError:
            return torch.zeros((2, 0), dtype=torch.long)

        if df.empty:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(df[["src", "dst"]].values.T, dtype=torch.long)

    def reverse(ei):
        return ei.flip(0)

    resolves_to = edge_index("edges_resolves_to.csv")
    shares_ns = edge_index("edges_shares_ns.csv")
    registered_by = edge_index("edges_registered_by.csv")
    belongs_asn = edge_index("edges_belongs_asn.csv")

    # Forward edges (semantic direction, kept for interpretability/paper description)
    data["domain", "resolves_to", "ip"].edge_index = resolves_to
    data["domain", "shares_nameserver", "nameserver"].edge_index = shares_ns
    data["domain", "registered_by", "registrar"].edge_index = registered_by
    data["ip", "belongs_to_asn", "asn"].edge_index = belongs_asn

    # Reverse edges (required so information actually flows back to Domain)
    data["ip", "rev_resolves_to", "domain"].edge_index = reverse(resolves_to)
    data["nameserver", "rev_shares_nameserver", "domain"].edge_index = reverse(shares_ns)
    data["registrar", "rev_registered_by", "domain"].edge_index = reverse(registered_by)
    data["asn", "rev_belongs_to_asn", "ip"].edge_index = reverse(belongs_asn)

    torch.save(data, f"data_processed/graphs/{snap_dir.name}.pt")
    print(snap_dir.name, "saved")

domains = pd.read_csv("data_processed/enriched/domains.csv")
import json
domain_id = json.load(open("data_processed/id_maps/domain_id_map.json"))
domains["node_id"] = domains["domain"].map(domain_id)
label_map = {"benign": 0, "malicious": 1}
domains["y"] = domains["label"].map(label_map)
domains[["node_id", "y"]].drop_duplicates("node_id").sort_values("node_id").to_csv(
    "data_processed/graphs/labels.csv", index=False)
print("Labels saved")