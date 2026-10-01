import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, csv
sys.path.append(os.getcwd())
from models.full_model import FullModel
from sklearn.metrics import f1_score, roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

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

MAX_EPOCHS = 1000
PATIENCE = 10

Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
Path("results/logs").mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
summary_rows = []

for seed in SEEDS:
    print(f"\n=== Training seed {seed} ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = FullModel(hidden_dim=64, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1
    epochs_no_improve = 0

    log_path = f"results/logs/train_log_seed{seed}.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_f1", "val_auc"])

    epoch = 0
    for epoch in range(MAX_EPOCHS):
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
            val_probs = torch.softmax(out_val[val_idx], dim=1)[:, 1].cpu().numpy()
            val_preds = out_val[val_idx].argmax(dim=1).cpu().numpy()
            val_true = y[val_idx].cpu().numpy()
            val_f1 = f1_score(val_true, val_preds, zero_division=0)
            try:
                val_auc = roc_auc_score(val_true, val_probs)
            except ValueError:
                val_auc = float("nan")

        if epoch % 20 == 0:
            print(f"epoch {epoch}: train_loss={train_loss.item():.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss.item(), val_loss, val_f1, val_auc])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_val_f1": best_val_f1,
            }, f"models/checkpoints/best_model_seed{seed}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}, best val F1 = {best_val_f1:.4f}")
                break

    summary_rows.append({"seed": seed, "best_val_f1": best_val_f1, "epochs_trained": epoch + 1})

pd.DataFrame(summary_rows).to_csv("results/logs/multiseed_summary.csv", index=False)
print("\nAll seeds complete:")
print(pd.DataFrame(summary_rows))