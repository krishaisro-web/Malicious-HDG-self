import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.getcwd())
from models.full_model import FullModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

snapshot_files = sorted(Path("data_processed/graphs").glob("snapshot_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
snapshots = [torch.load(f, weights_only=False).to(DEVICE) for f in snapshot_files]

labels = pd.read_csv("data_processed/graphs/labels.csv").sort_values("node_id")
y = torch.tensor(labels["y"].values, dtype=torch.long).to(DEVICE)
label_lookup = labels.set_index("node_id")["y"].to_dict()

snap_assign = pd.read_csv("data_processed/graphs/domain_snapshot_id.csv").sort_values("node_id")
domain_snapshot_id = torch.tensor(snap_assign["snapshot_id"].values, dtype=torch.long).to(DEVICE)

train_pool_ids = snap_assign[snap_assign["snapshot_id"] <= 18]["node_id"].tolist()
test_ids = snap_assign[snap_assign["snapshot_id"] == 19]["node_id"].tolist()
train_labels = [label_lookup[i] for i in train_pool_ids]

results = []
for seed in [42, 123, 2024]:
    train_ids, val_ids = train_test_split(train_pool_ids, test_size=0.15, stratify=train_labels, random_state=seed)
    train_idx = torch.tensor(train_ids, dtype=torch.long).to(DEVICE)
    val_idx = torch.tensor(val_ids, dtype=torch.long).to(DEVICE)
    test_idx = torch.tensor(test_ids, dtype=torch.long).to(DEVICE)

    torch.manual_seed(seed)
    model = FullModel(hidden_dim=64, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_val_f1, epochs_no_improve, best_state = -1, 0, None
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(snapshots, domain_snapshot_id)
        loss = loss_fn(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(snapshots, domain_snapshot_id)
            val_f1 = f1_score(y[val_idx].cpu().numpy(), out_val[val_idx].argmax(1).cpu().numpy(), zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1, epochs_no_improve = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 30:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_test = model(snapshots, domain_snapshot_id)
        test_probs = torch.softmax(out_test[test_idx], dim=1)[:, 1].cpu().numpy()
        test_preds = out_test[test_idx].argmax(dim=1).cpu().numpy()
        test_true = y[test_idx].cpu().numpy()

    test_f1 = f1_score(test_true, test_preds, zero_division=0)
    test_auc = roc_auc_score(test_true, test_probs) if len(set(test_true)) > 1 else float("nan")
    results.append({"seed": seed, "best_val_f1": best_val_f1, "test_f1": test_f1, "test_auc": test_auc})
    print(results[-1])

pd.DataFrame(results).to_csv("results/tables/window_18_19_seed_check.csv", index=False)
print("\nSummary:")
print(pd.DataFrame(results))
