import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.getcwd())
from models.full_model import FullModel
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

snapshot_files = sorted(Path("data_processed/graphs").glob("snapshot_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
snapshots = [torch.load(f, weights_only=False).to(DEVICE) for f in snapshot_files]

labels = pd.read_csv("data_processed/graphs/labels.csv").sort_values("node_id")
y = torch.tensor(labels["y"].values, dtype=torch.long).to(DEVICE)

snap_assign = pd.read_csv("data_processed/graphs/domain_snapshot_id.csv").sort_values("node_id")
domain_snapshot_id = torch.tensor(snap_assign["snapshot_id"].values, dtype=torch.long).to(DEVICE)

def load_split(name):
    with open(f"data_processed/graphs/split_{name}.txt") as f:
        return [int(x) for x in f.read().split()]

train_idx = torch.tensor(load_split("train"), dtype=torch.long).to(DEVICE)
val_idx = torch.tensor(load_split("val"), dtype=torch.long).to(DEVICE)
test_idx = torch.tensor(load_split("test"), dtype=torch.long).to(DEVICE)

VARIANTS = {"domain_ip": {("domain", "resolves_to", "ip"), ("ip", "rev_resolves_to", "domain")}}
VARIANTS["domain_ip_ns"] = VARIANTS["domain_ip"] | {
    ("domain", "shares_nameserver", "nameserver"), ("nameserver", "rev_shares_nameserver", "domain")}
VARIANTS["domain_ip_ns_reg"] = VARIANTS["domain_ip_ns"] | {
    ("domain", "registered_by", "registrar"), ("registrar", "rev_registered_by", "domain")}
VARIANTS["full_model"] = VARIANTS["domain_ip_ns_reg"] | {
    ("ip", "belongs_to_asn", "asn"), ("asn", "rev_belongs_to_asn", "ip")}

MAX_EPOCHS = 100
PATIENCE = 10
results = []

for variant_name, relations in VARIANTS.items():
    print(f"\n=== Training variant: {variant_name} ===")
    torch.manual_seed(42)
    model = FullModel(hidden_dim=64, dropout=0.3, relations=relations).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1
    epochs_no_improve = 0
    best_state = None
    epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(snapshots, domain_snapshot_id)
        loss = loss_fn(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(snapshots, domain_snapshot_id)
            val_preds = out_val[val_idx].argmax(dim=1).cpu().numpy()
            val_true = y[val_idx].cpu().numpy()
            val_f1 = f1_score(val_true, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_test = model(snapshots, domain_snapshot_id)
        test_probs = torch.softmax(out_test[test_idx], dim=1)[:, 1].cpu().numpy()
        test_preds = out_test[test_idx].argmax(dim=1).cpu().numpy()
        test_true = y[test_idx].cpu().numpy()

    metrics = {
        "variant": variant_name,
        "accuracy": accuracy_score(test_true, test_preds),
        "precision": precision_score(test_true, test_preds, zero_division=0),
        "recall": recall_score(test_true, test_preds, zero_division=0),
        "f1": f1_score(test_true, test_preds, zero_division=0),
        "roc_auc": roc_auc_score(test_true, test_probs),
        "best_val_f1": best_val_f1,
        "epochs_trained": epoch + 1,
    }
    print(metrics)
    results.append(metrics)

Path("results/tables").mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv("results/tables/ablation_summary.csv", index=False)
print("\nAblation complete.")