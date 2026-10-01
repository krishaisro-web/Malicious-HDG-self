import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os, json
sys.path.append(os.getcwd())
from models.full_model import FullModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score, confusion_matrix,
                              roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

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
test_idx_list = load_split("test")
test_idx = torch.tensor(test_idx_list, dtype=torch.long).to(DEVICE)

model = FullModel(hidden_dim=64, dropout=0.3).to(DEVICE)
checkpoint = torch.load(f"models/checkpoints/best_model_seed{SEED}.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    out = model(snapshots, domain_snapshot_id)
    probs = torch.softmax(out[test_idx], dim=1)[:, 1].cpu().numpy()
    preds = out[test_idx].argmax(dim=1).cpu().numpy()
    true = y[test_idx].cpu().numpy()

acc = accuracy_score(true, preds)
prec = precision_score(true, preds, zero_division=0)
rec = recall_score(true, preds, zero_division=0)
f1 = f1_score(true, preds, zero_division=0)
auc = roc_auc_score(true, probs)
pr_auc = average_precision_score(true, probs)
cm = confusion_matrix(true, preds)
tn, fp, fn, tp = cm.ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
fnr_val = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
tpr_val = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
tnr_val = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

metrics = {
    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
    "roc_auc": auc, "pr_auc": pr_auc,
    "true_positives": int(tp), "true_negatives": int(tn),
    "false_positives": int(fp), "false_negatives": int(fn),
    "false_positive_rate": fpr_val, "false_negative_rate": fnr_val,
    "true_positive_rate": tpr_val, "true_negative_rate": tnr_val,
}

Path("results/tables").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

with open(f"results/tables/test_metrics_seed{SEED}.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

pred_df = pd.DataFrame({
    "node_id": test_idx_list,
    "true_label": true,
    "predicted_label": preds,
    "predicted_prob_malicious": probs,
})
pred_df.to_csv(f"results/tables/test_predictions_seed{SEED}.csv", index=False)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig(f"results/figures/confusion_matrix_seed{SEED}.png", dpi=150); plt.close()

fpr_arr, tpr_arr, _ = roc_curve(true, probs)
plt.figure(figsize=(5, 4))
plt.plot(fpr_arr, tpr_arr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
plt.legend(); plt.tight_layout(); plt.savefig(f"results/figures/roc_curve_seed{SEED}.png", dpi=150); plt.close()

prec_arr, rec_arr, _ = precision_recall_curve(true, probs)
plt.figure(figsize=(5, 4))
plt.plot(rec_arr, prec_arr, label=f"PR-AUC = {pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
plt.legend(); plt.tight_layout(); plt.savefig(f"results/figures/pr_curve_seed{SEED}.png", dpi=150); plt.close()

# Per-malware-type breakdown (bonus, uses malware_type field)
domains_full = pd.read_csv("data_processed/enriched/domains.csv")
domain_id_map = json.load(open("data_processed/id_maps/domain_id_map.json"))
domains_full["node_id"] = domains_full["domain"].map(domain_id_map)
type_lookup = domains_full.set_index("node_id")["malware_type"].to_dict()

breakdown_rows = []
for nid, t, p in zip(test_idx_list, true, preds):
    if t == 1:
        breakdown_rows.append({"node_id": nid, "malware_type": type_lookup.get(nid, "unknown"), "correct": int(t == p)})
breakdown_df = pd.DataFrame(breakdown_rows)
if len(breakdown_df) > 0:
    per_type = breakdown_df.groupby("malware_type")["correct"].agg(["mean", "count"]).reset_index()
    per_type.columns = ["malware_type", "detection_rate", "count"]
    per_type.to_csv("results/tables/per_malware_type_breakdown.csv", index=False)
    print(per_type)

print("Evaluation complete.")