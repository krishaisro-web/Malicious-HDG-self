import pandas as pd, json
from pathlib import Path


node_counts = {}
for nt in ["domain", "ip", "nameserver", "registrar", "asn"]:
    m = json.load(open(f"data_processed/id_maps/{nt}_id_map.json"))
    node_counts[nt] = len(m)


edge_counts = {}
snap_dirs = sorted(
    [p for p in Path("data_processed/graphs").glob("snapshot_*") if p.is_dir()],
    key=lambda p: int(p.name.split("_")[1])
)
for etype in ["resolves_to", "shares_ns", "registered_by", "belongs_asn"]:
    total = 0
    for d in snap_dirs:
        f = d / f"edges_{etype}.csv"
        if f.exists() and f.stat().st_size > 0:
            try:
                total += len(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    edge_counts[etype] = total


print("Total nodes:", sum(node_counts.values()), node_counts)
print("Total edges:", sum(edge_counts.values()), edge_counts)