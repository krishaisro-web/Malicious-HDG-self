import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import sys, os
sys.path.append(os.getcwd())
from models.full_model import FullModel

torch.manual_seed(42)

snapshot_files = sorted(Path("data_processed/graphs").glob("snapshot_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
snapshots = [torch.load(f, weights_only=False) for f in snapshot_files]
labels = pd.read_csv("data_processed/graphs/labels.csv")
y = torch.tensor(labels["y"].values, dtype=torch.long)
domain_snapshot_id = torch.tensor(pd.read_csv("data_processed/graphs/domain_snapshot_id.csv")["snapshot_id"].values, dtype=torch.long)

model = FullModel(hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    optimizer.zero_grad()
    out = model(snapshots, domain_snapshot_id)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    print(f"epoch {epoch}: loss = {loss.item():.4f}")

print("Dry run complete — no errors.")