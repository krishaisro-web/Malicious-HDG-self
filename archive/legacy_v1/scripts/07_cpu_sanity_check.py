import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, csv
sys.path.append(os.getcwd())
from models.full_model import FullModel
from sklearn.metrics import f1_score, roc_auc_score

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")
print("Sanity check running on:", DEVICE)

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

model = FullModel(hidden_dim=64, dropout=0.3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

SANITY_EPOCHS = 20
Path("results/logs").mkdir(parents=True, exist_ok=True)
log_path = "results/logs/cpu_sanity_log.csv"

with open(log_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_auc"])

for epoch in range(SANITY_EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(snapshots, domain_snapshot_id)
    train_loss = loss_fn(out[train_idx], y[train_idx])
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val = model(snapshots, domain_snapshot_id)
        val_loss = loss_fn(out_val[val_idx], y[val_idx]).item()
        val_probs = torch.softmax(out_val[val_idx], dim=1)[:, 1].numpy()
        val_preds = out_val[val_idx].argmax(dim=1).numpy()
        val_true = y[val_idx].numpy()
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_true, val_probs)
        except ValueError:
            val_auc = float("nan")

    print(f"epoch {epoch}: train_loss={train_loss.item():.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}")
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch, train_loss.item(), val_loss, val_f1, val_auc])

print("CPU sanity check complete.")