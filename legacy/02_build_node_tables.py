import pandas as pd
import json
from pathlib import Path
import ast

Path("data_processed/id_maps").mkdir(parents=True, exist_ok=True)
Path("data_processed/graphs").mkdir(parents=True, exist_ok=True)

def parse_list_field(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

domains = pd.read_csv("data_processed/enriched/domains.csv")
dns = pd.read_csv("data_processed/enriched/dns.csv")
whois = pd.read_csv("data_processed/enriched/whois.csv")
asn = pd.read_csv("data_processed/enriched/asn.csv")

dns["a_records"] = dns["a_records"].apply(parse_list_field)
dns["aaaa_records"] = dns["aaaa_records"].apply(parse_list_field)
dns["ns_records"] = dns["ns_records"].apply(parse_list_field)

def save_id_map(values, name):
    # Force every value to string BEFORE building the map, so in-memory dict
    # and the JSON file on disk always agree on key type.
    unique_sorted = sorted(set(str(v) for v in values))
    id_map = {v: i for i, v in enumerate(unique_sorted)}
    with open(f"data_processed/id_maps/{name}_id_map.json", "w") as f:
        json.dump(id_map, f)
    return id_map

domain_id_map = save_id_map(domains["domain"], "domain")
common_tlds = {"com", "net", "org", "info"}

def tld_bucket(d):
    tld = d.split(".")[-1] if "." in d else ""
    return tld if tld in common_tlds else "other"

domain_feats = []
nx_lookup = dns.set_index("domain")["nxdomain"].to_dict()
tls_lookup = domains.set_index("domain")["has_tls"].to_dict()
for d, did in domain_id_map.items():
    tld = tld_bucket(d)
    domain_feats.append({
        "node_id": did,
        "domain": d,
        "length": len(d),
        "num_labels": d.count(".") + 1,
        "tld_com": int(tld == "com"),
        "tld_net": int(tld == "net"),
        "tld_org": int(tld == "org"),
        "tld_other": int(tld == "other"),
        "nxdomain": int(nx_lookup.get(d, False)),
        "has_tls": int(tls_lookup.get(d, 0)),
    })
pd.DataFrame(domain_feats).to_csv("data_processed/graphs/nodes_domain.csv", index=False)

ip_fanin = {}
for _, row in dns.iterrows():
    for ip in row["a_records"] + row["aaaa_records"]:
        ip_fanin[str(ip)] = ip_fanin.get(str(ip), 0) + 1
ip_id_map = save_id_map(ip_fanin.keys(), "ip")
ip_feats = [{"node_id": iid, "ip": ip, "ip_version": 6 if ":" in ip else 4, "fanin": ip_fanin[ip]} for ip, iid in ip_id_map.items()]
pd.DataFrame(ip_feats).to_csv("data_processed/graphs/nodes_ip.csv", index=False)

ns_fanin = {}
for _, row in dns.iterrows():
    for ns in row["ns_records"]:
        ns_fanin[str(ns)] = ns_fanin.get(str(ns), 0) + 1
ns_id_map = save_id_map(ns_fanin.keys(), "nameserver")
ns_feats = [{"node_id": nid, "nameserver": ns, "fanin": ns_fanin[ns]} for ns, nid in ns_id_map.items()]
pd.DataFrame(ns_feats).to_csv("data_processed/graphs/nodes_nameserver.csv", index=False)

reg_counts = {}
for r in whois.dropna(subset=["registrar"])["registrar"]:
    reg_counts[str(r)] = reg_counts.get(str(r), 0) + 1
reg_id_map = save_id_map(reg_counts.keys(), "registrar")
reg_feats = [{"node_id": rid, "registrar": r, "fanin": reg_counts[r]} for r, rid in reg_id_map.items()]
pd.DataFrame(reg_feats).to_csv("data_processed/graphs/nodes_registrar.csv", index=False)

asn_counts = {}
for a in asn.dropna(subset=["asn"])["asn"]:
    asn_counts[str(a)] = asn_counts.get(str(a), 0) + 1
asn_id_map = save_id_map(asn_counts.keys(), "asn")
asn_feats = [{"node_id": aid, "asn": a, "fanin": asn_counts[a]} for a, aid in asn_id_map.items()]
pd.DataFrame(asn_feats).to_csv("data_processed/graphs/nodes_asn.csv", index=False)

print("Node tables built:", len(domain_id_map), "domains,", len(ip_id_map), "IPs,",
      len(ns_id_map), "nameservers,", len(reg_id_map), "registrars,", len(asn_id_map), "ASNs")