import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

labels = pd.read_csv("data_processed/graphs/labels.csv").sort_values("node_id")
node_ids = labels["node_id"].values
y = labels["y"].values

train_ids, temp_ids, y_train, y_temp = train_test_split(
    node_ids, y, test_size=0.30, stratify=y, random_state=42
)
val_ids, test_ids, y_val, y_test = train_test_split(
    temp_ids, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

Path("data_processed/graphs").mkdir(parents=True, exist_ok=True)
for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    with open(f"data_processed/graphs/split_{name}.txt", "w") as f:
        f.write("\n".join(map(str, sorted(ids))))
    print(name, len(ids))