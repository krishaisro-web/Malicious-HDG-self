import ijson
import pandas as pd
from pathlib import Path
import random

def extract_ns(r):
    rdap = r.get("rdap") or {}
    ns_list = rdap.get("nameservers") or []
    return sorted(set(ns.lower().strip() for ns in ns_list if ns))

def extract_registrar(r):
    rdap = r.get("rdap") or {}
    entities = rdap.get("entities") or {}
    registrar_entities = entities.get("registrar") or []
    if registrar_entities:
        first = registrar_entities[0]
        return first.get("name") or first.get("handle")
    return None

def extract_date(date_field):
    if isinstance(date_field, dict):
        return date_field.get("$date")
    return date_field


def parse_file(path, label, sample_size, seed=42):
    random.seed(seed)
    domain_rows, dns_rows, whois_rows, asn_rows = [], [], [], []
    reservoir = []
    with open(path, "rb") as f:
        for i, r in enumerate(ijson.items(f, "item")):
            if i < sample_size:
                reservoir.append(r)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = r
    for r in reservoir:
        domain = (r.get("domain_name") or "").lower().strip()
        if not domain:
            continue
        dns = r.get("dns") or {}
        rdap = r.get("rdap") or {}
        ip_data = r.get("ip_data") or []

        a_records = dns.get("A") or []
        aaaa_records = dns.get("AAAA") or []

        domain_rows.append({
            "domain": domain,
            "label": label,
            "malware_type": r.get("malware_type"),
            "evaluated_on": extract_date(r.get("evaluated_on")),
            "has_tls": int(bool(r.get("tls"))),
        })
        dns_rows.append({
            "domain": domain,
            "a_records": a_records,
            "aaaa_records": aaaa_records,
            "ns_records": extract_ns(r),
            "nxdomain": (len(a_records) == 0 and len(aaaa_records) == 0),
        })
        whois_rows.append({
            "domain": domain,
            "registrar": extract_registrar(r),
            "registration_date": extract_date(rdap.get("registration_date")),
        })
        for ip_entry in ip_data:
            asn_info = ip_entry.get("asn") or {}
            asn_rows.append({
                "ip": ip_entry.get("ip"),
                "asn": asn_info.get("asn"),
                "asn_org": asn_info.get("as_org"),
            })
            
    return domain_rows, dns_rows, whois_rows, asn_rows


SAMPLE_MALICIOUS = 15000
SAMPLE_BENIGN = 15000

mal = parse_file("data_raw/zenodo/malware.json", "malicious", SAMPLE_MALICIOUS)
ben = parse_file("data_raw/zenodo/benign_umbrella.json", "benign", SAMPLE_BENIGN)

Path("data_processed/enriched").mkdir(parents=True, exist_ok=True)

for name, mal_rows, ben_rows in zip(["domains", "dns", "whois", "asn"], mal, ben):
    df = pd.DataFrame(mal_rows + ben_rows)
    if name == "asn":
        df = df.dropna(subset=["ip"]).drop_duplicates(subset="ip")
    df.to_csv(f"data_processed/enriched/{name}.csv", index=False)
    print(name, "->", len(df), "rows")