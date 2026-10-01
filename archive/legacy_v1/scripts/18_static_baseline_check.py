# src/scripts/18_static_baseline_check.py
import torch, torch.nn as nn, pandas as pd, numpy as np
from pathlib import Path
from torch_geometric.data import HeteroData
import sys, os
sys.path.append(os.getcwd())
from models.encoder import FeatureProjection, make_hetero_layer, ALL_RELATIONS
from models.classifier import Classifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_node_features(name, feature_cols, log_scale_cols=None):
    df = pd.read_csv(f"data_processed/graphs/nodes_{name}.csv").sort_values("node_id")
    log_scale_cols = log_scale_cols or []
    data = df[feature_cols].copy().astype(float)
    for col in log_scale_cols:
        data[col] = torch.log1p(torch.tensor(data[col].values, dtype=torch.float)).numpy()
    for col in data.columns:
        std, mean = data[col].std(), data[col].mean()
        data[col] = (data[col] - mean) / std if std > 1e-8 else data[col] - mean
    return torch.tensor(data.values, dtype=torch.float)

# --- Merge all snapshots' edges into one static graph ---
snap_dirs = sorted([p for p in Path("data_processed/graphs").glob("snapshot_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[1]))

def merge_edges(fname):
    dfs = []
    for d in snap_dirs:
        try:
            df = pd.read_csv(d / fname)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return torch.zeros((2, 0), dtype=torch.long)
    merged = pd.concat(dfs).drop_duplicates()
    return torch.tensor(merged[["src", "dst"]].values.T, dtype=torch.long)

resolves_to = merge_edges("edges_resolves_to.csv")
shares_ns = merge_edges("edges_shares_ns.csv")
registered_by = merge_edges("edges_registered_by.csv")
belongs_asn = merge_edges("edges_belongs_asn.csv")

data = HeteroData()
data["domain"].x = load_node_features("domain", ["length", "num_labels", "tld_com", "tld_net", "tld_org", "tld_other", "nxdomain", "has_tls"])
data["ip"].x = load_node_features("ip", ["ip_version", "fanin"], log_scale_cols=["fanin"])
data["nameserver"].x = load_node_features("nameserver", ["fanin"], log_scale_cols=["fanin"])
data["registrar"].x = load_node_features("registrar", ["fanin"], log_scale_cols=["fanin"])
data["asn"].x = load_node_features("asn", ["fanin"], log_scale_cols=["fanin"])
data["domain", "resolves_to", "ip"].edge_index = resolves_to
data["domain", "shares_nameserver", "nameserver"].edge_index = shares_ns
data["domain", "registered_by", "registrar"].edge_index = registered_by
data["ip", "belongs_to_asn", "asn"].edge_index = belongs_asn
data["ip", "rev_resolves_to", "domain"].edge_index = resolves_to.flip(0)
data["nameserver", "rev_shares_nameserver", "domain"].edge_index = shares_ns.flip(0)
data["registrar", "rev_registered_by", "domain"].edge_index = registered_by.flip(0)
data["asn", "rev_belongs_to_asn", "ip"].edge_index = belongs_asn.flip(0)
data = data.to(DEVICE)

print(f"Merged static graph — domain-ip edges: {resolves_to.shape[1]}, "
      f"domain-ns: {shares_ns.shape[1]}, domain-reg: {registered_by.shape[1]}, ip-asn: {belongs_asn.shape[1]}")

# --- Same model architecture, single-graph forward pass (no snapshot gather) ---
class StaticModel(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.proj = FeatureProjection(hidden_dim)
        self.layer1 = make_hetero_layer(hidden_dim, ALL_RELATIONS)
        self.layer2 = make_hetero_layer(hidden_dim, ALL_RELATIONS)
        self.dropout = nn.Dropout(dropout)
        self.classifier = Classifier(hidden_dim)

    def forward(self, data):
        x_dict = self.proj(data.x_dict)
        x_dict = self.layer1(x_dict, data.edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.layer2(x_dict, data.edge_index_dict)
        x_dict = {k: self.dropout(torch.relu(v)) for k, v in x_dict.items()}
        return self.classifier(x_dict["domain"])

labels = pd.read_csv("data_processed/graphs/labels.csv").sort_values("node_id")
y = torch.tensor(labels["y"].values, dtype=torch.long).to(DEVICE)

def load_split(name):
    with open(f"data_processed/graphs/split_{name}.txt") as f:
        return [int(x) for x in f.read().split()]
train_idx = torch.tensor(load_split("train"), dtype=torch.long).to(DEVICE)
val_idx = torch.tensor(load_split("val"), dtype=torch.long).to(DEVICE)
test_idx = torch.tensor(load_split("test"), dtype=torch.long).to(DEVICE)

torch.manual_seed(42)
model = StaticModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

best_val_f1, epochs_no_improve, best_state = -1, 0, None
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val = model(data)
        val_f1 = f1_score(y[val_idx].cpu().numpy(), out_val[val_idx].argmax(1).cpu().numpy(), zero_division=0)
    if val_f1 > best_val_f1:
        best_val_f1, epochs_no_improve = val_f1, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= 15:
            break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out_test = model(data)
    probs = torch.softmax(out_test[test_idx], dim=1)[:, 1].cpu().numpy()
    preds = out_test[test_idx].argmax(dim=1).cpu().numpy()
    true = y[test_idx].cpu().numpy()

print("\n=== STATIC MERGED GRAPH (no snapshot partitioning) ===")
print(f"F1: {f1_score(true, preds):.4f}")
print(f"AUC: {roc_auc_score(true, probs):.4f}")
print(f"Accuracy: {accuracy_score(true, preds):.4f}")
print(f"Precision: {precision_score(true, preds):.4f}, Recall: {recall_score(true, preds):.4f}")