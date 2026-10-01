import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())

from models.full_model import FullModel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SEED = 42
START_TRAIN_SNAPSHOT = 9
MIN_TEST_SIZE = 30

# Diagnostic mode: run exactly one temporal window first.
# Set to None later to run all valid windows.
DIAGNOSTIC_CUTOFF = None

MAX_EPOCHS = 100
PATIENCE = 10
VAL_FRACTION = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Load snapshot graphs
# ------------------------------------------------------------
snapshot_files = sorted(
    Path("data_processed/graphs").glob("snapshot_*.pt"),
    key=lambda p: int(p.stem.split("_")[1]),
)

NUM_SNAPSHOTS = len(snapshot_files)

if NUM_SNAPSHOTS < 2:
    raise RuntimeError("Need at least two graph snapshots.")

print(f"Device: {DEVICE}")
print(f"Available snapshots: 0..{NUM_SNAPSHOTS - 1}")


# ------------------------------------------------------------
# Load labels and temporal assignments
# ------------------------------------------------------------
labels = (
    pd.read_csv("data_processed/graphs/labels.csv")
    .sort_values("node_id")
    .reset_index(drop=True)
)

snap_assign = (
    pd.read_csv("data_processed/graphs/domain_snapshot_id.csv")
    .sort_values("node_id")
    .reset_index(drop=True)
)

if not labels["node_id"].equals(snap_assign["node_id"]):
    raise RuntimeError(
        "labels.csv and domain_snapshot_id.csv are not aligned by node_id."
    )

y_all = torch.tensor(labels["y"].values, dtype=torch.long)
domain_snapshot_id_all = torch.tensor(
    snap_assign["snapshot_id"].values,
    dtype=torch.long,
)


# ------------------------------------------------------------
# Determine temporal windows
# ------------------------------------------------------------
if DIAGNOSTIC_CUTOFF is not None:
    cutoffs = [DIAGNOSTIC_CUTOFF]
else:
    cutoffs = list(range(START_TRAIN_SNAPSHOT, NUM_SNAPSHOTS - 1))

results = []

for cutoff in cutoffs:
    test_snapshot = cutoff + 1

    if cutoff < START_TRAIN_SNAPSHOT:
        continue

    if test_snapshot >= NUM_SNAPSHOTS:
        continue

    # --------------------------------------------------------
    # Temporal domain selection
    # --------------------------------------------------------
    train_mask = domain_snapshot_id_all <= cutoff
    test_mask = domain_snapshot_id_all == test_snapshot

    train_idx_all = torch.nonzero(train_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]

    if len(test_idx) < MIN_TEST_SIZE:
        print(
            f"Skipping cutoff {cutoff}: "
            f"test snapshot {test_snapshot} has only {len(test_idx)} domains."
        )
        continue

    # --------------------------------------------------------
    # Stratified historical train/validation split
    # --------------------------------------------------------
    train_labels_np = y_all[train_idx_all].numpy()

    fit_pos, val_pos = train_test_split(
        np.arange(len(train_idx_all)),
        test_size=VAL_FRACTION,
        stratify=train_labels_np,
        random_state=SEED,
    )

    fit_idx = train_idx_all[fit_pos]
    val_idx = train_idx_all[val_pos]

    # --------------------------------------------------------
    # Restrict graph availability to historical + test snapshot
    # --------------------------------------------------------
    available_snapshot_ids = list(range(test_snapshot + 1))

    snapshots = [
        torch.load(
            snapshot_files[snap_id],
            weights_only=False,
        ).to(DEVICE)
        for snap_id in available_snapshot_ids
    ]

    # --------------------------------------------------------
    # Local snapshot assignments
    #
    # All selected domains have assignments inside
    # 0..test_snapshot, so the original FullModel gather
    # operation remains valid.
    # --------------------------------------------------------
    domain_snapshot_id = domain_snapshot_id_all.clone()

    # Domains after the current test snapshot are outside this
    # temporal experiment. Their outputs are never evaluated, but
    # FullModel still requires every assignment to reference a
    # snapshot that is currently loaded. Map excluded future domains
    # to snapshot 0 so they cannot create an invalid index.
    future_mask = domain_snapshot_id > test_snapshot
    domain_snapshot_id[future_mask] = 0

    # Sanity check: all domains used by this experiment now reference
    # snapshots that are actually available.
    selected_idx = torch.cat([train_idx_all, test_idx])
    max_selected_snapshot = int(
        domain_snapshot_id[selected_idx].max().item()
    )

    if max_selected_snapshot != test_snapshot:
        raise RuntimeError(
            f"Unexpected maximum selected snapshot: "
            f"{max_selected_snapshot}; expected {test_snapshot}."
        )

    # --------------------------------------------------------
    # Move tensors to device
    # --------------------------------------------------------
    y = y_all.to(DEVICE)
    domain_snapshot_id = domain_snapshot_id.to(DEVICE)
    fit_idx = fit_idx.to(DEVICE)
    val_idx = val_idx.to(DEVICE)
    test_idx = test_idx.to(DEVICE)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    set_seed(SEED)

    model = FullModel(hidden_dim=64, dropout=0.3).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = -1

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print(
        f"Temporal window: train snapshots 0..{cutoff}, "
        f"test snapshot {test_snapshot}"
    )
    print(f"Historical domains: {len(train_idx_all)}")
    print(f"Fit domains:        {len(fit_idx)}")
    print(f"Validation domains: {len(val_idx)}")
    print(f"Test domains:       {len(test_idx)}")
    print(f"Available graphs:   0..{test_snapshot}")
    print("=" * 70)

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()

        out = model(snapshots, domain_snapshot_id)
        loss = loss_fn(out[fit_idx], y[fit_idx])

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            out_val = model(snapshots, domain_snapshot_id)

        val_preds = out_val[val_idx].argmax(dim=1).cpu().numpy()
        val_true = y[val_idx].cpu().numpy()

        val_f1 = f1_score(
            val_true,
            val_preds,
            zero_division=0,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch}: "
                f"train_loss={loss.item():.4f} "
                f"val_f1={val_f1:.4f}"
            )

        if epochs_no_improve >= PATIENCE:
            break

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------
    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Test exactly once
    # --------------------------------------------------------
    with torch.no_grad():
        out_test = model(snapshots, domain_snapshot_id)

    test_probs = torch.softmax(out_test[test_idx], dim=1)[:, 1].cpu().numpy()
    test_preds = out_test[test_idx].argmax(dim=1).cpu().numpy()
    test_true = y[test_idx].cpu().numpy()

    test_f1 = f1_score(
        test_true,
        test_preds,
        zero_division=0,
    )

    if len(np.unique(test_true)) >= 2:
        test_auc = roc_auc_score(test_true, test_probs)
    else:
        test_auc = float("nan")

    result = {
        "seed": SEED,
        "train_up_to_snapshot": cutoff,
        "test_snapshot": test_snapshot,
        "train_domains": int(len(fit_idx)),
        "validation_domains": int(len(val_idx)),
        "test_domains": int(len(test_idx)),
        "test_benign": int((test_true == 0).sum()),
        "test_malicious": int((test_true == 1).sum()),
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "test_f1": float(test_f1),
        "test_roc_auc": float(test_auc),
    }

    print()
    print("Temporal test result:")
    print(result)

    results.append(result)


# ------------------------------------------------------------
# Save all rolling-origin results
# ------------------------------------------------------------
Path("results/tables").mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    "results/tables/rolling_origin_temporal_diagnostic.csv",
    index=False,
)

print()
print(
    "Saved: "
    "results/tables/rolling_origin_temporal_diagnostic.csv"
)
print(f"Saved {len(results)} rolling-origin windows.")
print("Rolling-origin diagnostic complete.")
