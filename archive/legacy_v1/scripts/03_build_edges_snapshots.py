import pandas as pd
import json
import ast
from pathlib import Path

def load_id_map(name):
    return json.load(open(f"data_processed/id_maps/{name}_id_map.json"))

def parse_list_field(x):
    if isinstance(x, list): return x
    if pd.isna(x): return []
    try: return ast.literal_eval(x)
    except Exception: return []

domains = pd.read_csv("data_processed/enriched/domains.csv")
dns = pd.read_csv("data_processed/enriched/dns.csv")
whois = pd.read_csv("data_processed/enriched/whois.csv")
asn = pd.read_csv("data_processed/enriched/asn.csv")

dns["a_records"] = dns["a_records"].apply(parse_list_field)
dns["aaaa_records"] = dns["aaaa_records"].apply(parse_list_field)
dns["ns_records"] = dns["ns_records"].apply(parse_list_field)

domain_id = load_id_map("domain")
ip_id = load_id_map("ip")
ns_id = load_id_map("nameserver")
reg_id = load_id_map("registrar")
asn_id = load_id_map("asn")

ip_to_asn = {}
for _, row in asn.dropna(subset=["asn"]).iterrows():
    ip_to_asn[str(row["ip"])] = str(row["asn"])

domains["evaluated_on"] = pd.to_datetime(domains["evaluated_on"], errors="coerce", utc=True)
domains["evaluated_on"] = domains["evaluated_on"].fillna(domains["evaluated_on"].min())
domains["week"] = domains["evaluated_on"].dt.to_period("W").astype(str)
weeks = sorted(domains["week"].unique())
week_to_snap = {w: i for i, w in enumerate(weeks)}
domains["snapshot_id"] = domains["week"].map(week_to_snap)

dns_lookup = dns.set_index("domain")
whois_lookup = whois.set_index("domain")

for snap_id, group in domains.groupby("snapshot_id"):
    out_dir = Path(f"data_processed/graphs/snapshot_{snap_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolves_to, shares_ns, registered_by, belongs_asn = [], [], [], []
    for d in group["domain"]:
        did = domain_id[str(d)]
        if d in dns_lookup.index:
            row = dns_lookup.loc[d]
            a_recs = row["a_records"] if isinstance(row["a_records"], list) else []
            aaaa_recs = row["aaaa_records"] if isinstance(row["aaaa_records"], list) else []
            ns_recs = row["ns_records"] if isinstance(row["ns_records"], list) else []
            for ip in a_recs + aaaa_recs:
                ip_str = str(ip)
                if ip_str in ip_id:
                    resolves_to.append({"src": did, "dst": ip_id[ip_str]})
                    a = ip_to_asn.get(ip_str)
                    if a is not None and a in asn_id:
                        belongs_asn.append({"src": ip_id[ip_str], "dst": asn_id[a]})
            for ns in ns_recs:
                ns_str = str(ns)
                if ns_str in ns_id:
                    shares_ns.append({"src": did, "dst": ns_id[ns_str]})
        if d in whois_lookup.index:
            r = whois_lookup.loc[d, "registrar"]

            if isinstance(r, pd.Series):
                r = r.iloc[0]

            if pd.notna(r) and str(r) in reg_id:
                registered_by.append({
                    "src": did,
                    "dst": reg_id[str(r)]
                })
    pd.DataFrame(resolves_to).drop_duplicates().to_csv(out_dir / "edges_resolves_to.csv", index=False)
    pd.DataFrame(shares_ns).drop_duplicates().to_csv(out_dir / "edges_shares_ns.csv", index=False)
    pd.DataFrame(registered_by).drop_duplicates().to_csv(out_dir / "edges_registered_by.csv", index=False)
    pd.DataFrame(belongs_asn).drop_duplicates().to_csv(out_dir / "edges_belongs_asn.csv", index=False)

domains["node_id"] = domains["domain"].astype(str).map(domain_id)
domains[["node_id", "snapshot_id"]].drop_duplicates("node_id").sort_values("node_id").to_csv(
    "data_processed/graphs/domain_snapshot_id.csv", index=False)
print("Saved domain_snapshot_id.csv")
print(f"Built {len(weeks)} snapshots")
