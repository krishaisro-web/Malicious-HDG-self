"""
Synthetic schema fixture generator for Zenodo DomainRadar v2.
Produces schema-faithful JSON files with intentional overlaps to prevent trivial separability.
Outputs strictly to data_fixture/zenodo/.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_mongo_date(year: int, month: int, day: int, style: str = "iso") -> Any:
    """Generates dates in mongoexport formats: ISO string or numberLong timestamp."""
    iso_str = f"{year:04d}-{month:02d}-{day:02d}T12:00:00Z"
    if style == "numberLong":
        # Approximate millisecond timestamp
        ms = int(1672531200000 + (year - 2023) * 31536000000 + (month - 1) * 2592000000 + day * 86400000)
        return {"$date": {"$numberLong": str(ms)}}
    elif style == "iso_obj":
        return {"$date": iso_str}
    elif style == "plain_str":
        return iso_str
    else:
        return {"$date": iso_str}


def make_fixture(
    output_dir: Path,
    n_malware: int = 3000,
    n_umbrella: int = 1500,
    n_cesnet: int = 1500,
    seed: int = 42
) -> Dict[str, Path]:
    """
    Generates synthetic schema-faithful dataset files mirroring Zenodo DomainRadar v2:
    - malware.json
    - benign_umbrella.json
    - benign_cesnet.json
    - data_schema.json
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common vocabulary to ensure overlapping distributions
    tlds = [".com", ".net", ".org", ".info", ".biz", ".io"]
    tld_weights = [0.50, 0.15, 0.15, 0.08, 0.06, 0.06]

    subdomains = ["www", "api", "cdn", "mail", "app", "login", "auth", "dev"]
    stem_words = [
        "cloud", "secure", "connect", "portal", "system", "host", "media", "service",
        "matrix", "vector", "delta", "alpha", "net", "web", "link", "core", "pulse",
        "stream", "sync", "flow", "nexus", "vertex", "prime", "edge", "vault", "orbit"
    ]

    malware_sources = ["ThreatFox", "URLhaus", "Firebog", "MISP", "abuse.ch"]
    malware_families = ["emotet", "qakbot", "redline", "formbook", "asyncrat", "agenttesla", "unknown"]
    
    registrars = [
        ("GoDaddy.com, LLC", "146", "GoDaddy"),
        ("Namecheap, Inc.", "1068", "Namecheap"),
        ("Tucows Domains Inc.", "69", "Tucows"),
        ("Cloudflare, Inc.", "1910", "Cloudflare"),
        ("MarkMonitor Inc.", "292", "MarkMonitor"),
        ("Hostinger Operations, UAB", "1636", "Hostinger")
    ]

    asns = [
        (13335, "CLOUDFLARENET", "Cloudflare, Inc."),
        (15169, "GOOGLE", "Google LLC"),
        (16509, "AMAZON-02", "Amazon.com, Inc."),
        (20940, "AKAMAI-ASN1", "Akamai Technologies, Inc."),
        (14061, "DIGITALOCEAN-ASN", "DigitalOcean, LLC"),
        (24940, "HETZNER-AS", "Hetzner Online GmbH")
    ]

    nameservers = [
        ("ns1.cloudflare.com", "172.64.32.1"),
        ("ns2.cloudflare.com", "172.64.33.1"),
        ("dns1.registrar-servers.com", "198.54.117.10"),
        ("dns2.registrar-servers.com", "198.54.117.11"),
        ("ns1.domaincontrol.com", "97.74.100.1"),
        ("ns2.domaincontrol.com", "173.201.100.1"),
        ("ns-cloud-a1.googledomains.com", "216.239.32.106"),
        ("ns-112.awsdns-14.com", "205.251.192.112")
    ]

    months = [(2023, m) for m in range(4, 13)] + [(2024, m) for m in range(1, 8)]

    def generate_domain(idx: int, is_malware: bool) -> Dict[str, Any]:
        tld = rng.choices(tlds, weights=tld_weights, k=1)[0]
        stem = rng.choice(stem_words) + rng.choice(stem_words) + str(rng.randint(10, 99))
        has_sub = rng.random() < 0.28
        sub = rng.choice(subdomains) if has_sub else ""
        domain_name = f"{sub}.{stem}{tld}" if has_sub else f"{stem}{tld}"

        # Temporal assignment (sampled from the same monthly windows for both classes)
        yr, mo = rng.choice(months)
        day = rng.randint(1, 28)
        date_style = rng.choice(["iso_obj", "numberLong", "plain_str"])
        
        # In rare cases, simulate missing sourced_on to test fallback
        has_sourced_on = rng.random() > 0.05
        has_eval_on = rng.random() > 0.02
        
        sourced_on = generate_mongo_date(yr, mo, day, date_style) if has_sourced_on else None
        evaluated_on = generate_mongo_date(yr, mo, min(day + 2, 28), "iso_obj") if has_eval_on else None
        
        # Registrar
        reg_info = rng.choice(registrars)
        rdap_null = rng.random() < 0.08
        if rdap_null:
            rdap = None
        else:
            rdap = {
                "registration_date": generate_mongo_date(yr - rng.randint(0, 3), rng.randint(1, 12), rng.randint(1, 28), "iso_obj"),
                "expiration_date": generate_mongo_date(yr + rng.randint(1, 2), rng.randint(1, 12), rng.randint(1, 28), "iso_obj"),
                "last_changed_date": generate_mongo_date(yr, mo, day, "iso_obj"),
                "nameservers": [rng.choice(nameservers)[0], rng.choice(nameservers)[0]],
                "status": ["clientTransferProhibited", "active"],
                "dnssec": False,
                "entities": {
                    "registrar": [{
                        "name": reg_info[0],
                        "handle": reg_info[1],
                        "organization": reg_info[2]
                    }],
                    "abuse": [{"email": f"abuse@{reg_info[2].lower().replace(' ', '')}.com"}]
                }
            }

        # DNS
        dns_null = rng.random() < 0.05
        if dns_null:
            dns = None
            no_ips = True
            selected_ips = []
        else:
            no_ips = rng.random() < 0.08
            ip_val = f"198.51.{rng.randint(1, 250)}.{rng.randint(1, 250)}"
            selected_ips = [] if no_ips else [ip_val]
            
            ns_choice = rng.sample(nameservers, k=2)
            ns_dict = {
                ns[0]: {"related_ips": [{"ttl": 86400, "value": ns[1]}]}
                for ns in ns_choice
            }

            dns = {
                "A": selected_ips if selected_ips else None,
                "AAAA": None,
                "NS": ns_dict,
                "MX": {
                    f"mail.{domain_name}": {
                        "priority": 10,
                        "related_ips": [{"ttl": 3600, "value": f"198.51.100.{rng.randint(1, 250)}"}]
                    }
                } if rng.random() > 0.3 else None,
                "CNAME": None,
                "TXT": ["v=spf1 ~all"] if rng.random() > 0.5 else None,
                "SOA": {
                    "primary_ns": ns_choice[0][0],
                    "resp_mailbox_dname": f"hostmaster.{stem}{tld}",
                    "serial": 2023050101,
                    "refresh": 10000,
                    "retry": 2400,
                    "expire": 604800,
                    "min_ttl": 3600
                },
                "ttls": {"A": 300, "NS": 86400},
                "remarks": {
                    "has_dnskey": False,
                    "has_spf": rng.random() > 0.4,
                    "has_dmarc": rng.random() > 0.6,
                    "has_dkim": False,
                    "zone": tld.lstrip(".")
                },
                "dnssec": {"dnssec": "unsigned"}
            }

        # IP Data (includes IPs from A, NS, MX)
        ip_data_list = []
        asn_choice = rng.choice(asns)
        # Randomize ASN key structure to test robustness against candidate keys
        asn_key_variant = rng.choice(["number", "asn", "autonomous_system_number"])
        asn_obj = {
            asn_key_variant: asn_choice[0],
            "organization": asn_choice[2],
            "network": f"198.51.{rng.randint(1, 250)}.0/24"
        }

        for ip in selected_ips:
            ip_data_list.append({
                "ip": ip,
                "from_record": "A",
                "remarks": {"is_alive": True, "average_rtt": round(rng.uniform(10.0, 80.0), 2)},
                "rdap": {"handle": f"NET-{asn_choice[0]}"},
                "geo": {"country": "US", "city": "Ashburn"},
                "asn": asn_obj
            })
        
        # Also include related IP from NS or MX to verify from_record filtering
        if dns and dns.get("NS"):
            first_ns = list(dns["NS"].values())[0]
            ns_ip = first_ns["related_ips"][0]["value"]
            ip_data_list.append({
                "ip": ns_ip,
                "from_record": "NS",
                "remarks": {"is_alive": True},
                "geo": {"country": "US"},
                "asn": asn_obj
            })

        # TLS
        tls_null = rng.random() < 0.35
        if tls_null:
            tls = None
        else:
            # Certificates array with non-root (leaf) and root
            leaf_cert = {
                "common_name": domain_name,
                "organization": "Let's Encrypt" if rng.random() > 0.3 else "DigiCert Inc",
                "country": "US",
                "validity_start": generate_mongo_date(yr, mo, 1, "iso_obj"),
                "validity_end": generate_mongo_date(yr, min(mo + 3, 12), 1, "iso_obj"),
                "valid_len": 90,
                "extensions": [{"critical": False, "name": "basicConstraints", "value": "CA:FALSE"}],
                "extension_count": 1,
                "is_root": False
            }
            root_cert = {
                "common_name": "ISRG Root X1",
                "organization": "Internet Security Research Group",
                "country": "US",
                "validity_start": {"$date": "2015-06-04T11:04:38Z"},
                "validity_end": {"$date": "2035-06-04T11:04:38Z"},
                "valid_len": 7300,
                "extensions": [],
                "extension_count": 0,
                "is_root": True
            }
            # Test order independence: sometimes root first, sometimes leaf first
            certs = [root_cert, leaf_cert] if rng.random() > 0.5 else [leaf_cert, root_cert]
            tls = {
                "cipher": "TLS_AES_256_GCM_SHA384",
                "protocol": "TLSv1.3",
                "count": len(certs),
                "certificates": certs
            }

        rec: Dict[str, Any] = {
            "_id": {"$oid": f"{idx:024x}"},
            "domain_name": domain_name,
            "url": f"http://{domain_name}/",
            "sourced_on": sourced_on,
            "evaluated_on": evaluated_on,
            "dns": dns,
            "rdap": rdap,
            "tls": tls,
            "ip_data": ip_data_list if ip_data_list else None,
            "remarks": {
                "dns_evaluated_on": evaluated_on,
                "rdap_evaluated_on": evaluated_on,
                "tls_evaluated_on": evaluated_on,
                "dns_had_no_ips": len(selected_ips) == 0
            }
        }

        if is_malware:
            rec["source"] = rng.choice(malware_sources)
            fam = rng.choice(malware_families)
            # Test both key variants: 'malware' vs 'malware_type'
            if rng.random() > 0.5:
                rec["malware"] = fam
            else:
                rec["malware_type"] = fam
        else:
            rec["source"] = "Umbrella" if idx % 2 == 0 else "CESNET"

        return rec

    malware_records = [generate_domain(i, is_malware=True) for i in range(1, n_malware + 1)]
    umbrella_records = [generate_domain(100000 + i, is_malware=False) for i in range(1, n_umbrella + 1)]
    cesnet_records = [generate_domain(200000 + i, is_malware=False) for i in range(1, n_cesnet + 1)]

    # Schema definition file
    schema_def = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "DomainRadarRecord",
        "type": "object",
        "$defs": {
            "rdapEntity": {
                "type": "object",
                "description": "Entity structure inside RDAP object",
                "properties": {
                    "name": {"type": "string"},
                    "handle": {"type": "string"},
                    "organization": {"type": "string"},
                    "kind": {"type": "string"},
                    "roles": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }

    files = {
        "malware": output_dir / "malware.json",
        "umbrella": output_dir / "benign_umbrella.json",
        "cesnet": output_dir / "benign_cesnet.json",
        "schema": output_dir / "data_schema.json"
    }

    with open(files["malware"], "w", encoding="utf-8") as f:
        json.dump(malware_records, f, indent=None)

    with open(files["umbrella"], "w", encoding="utf-8") as f:
        json.dump(umbrella_records, f, indent=None)

    with open(files["cesnet"], "w", encoding="utf-8") as f:
        json.dump(cesnet_records, f, indent=None)

    with open(files["schema"], "w", encoding="utf-8") as f:
        json.dump(schema_def, f, indent=2)

    return files
