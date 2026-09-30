import json
import random
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

OUT = Path("data_raw/zenodo")
OUT.mkdir(parents=True, exist_ok=True)

def make_record(domain, label):
    day = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))

    ip = f"192.0.2.{random.randint(1,254)}"

    if label == "malicious":
        malware_type = "wordlist-dga"
    else:
        malware_type = None

    return {
        "domain_name": domain,
        "malware_type": malware_type,
        "evaluated_on": {"$date": day.isoformat() + "Z"},
        "tls": {
            "enabled": random.choice([True, False])
        },
        "dns": {
            "A": [ip],
            "AAAA": [],
        },
        "rdap": {
            "nameservers": [
                f"ns{random.randint(1,5)}.example.com"
            ],
            "entities": {
                "registrar": [
                    {
                        "name": random.choice(
                            [
                                "Example Registrar",
                                "Demo Registrar",
                                "Test Registrar"
                            ]
                        )
                    }
                ]
            },
            "registration_date": {
                "$date": day.isoformat() + "Z"
            }
        },
        "ip_data": [
            {
                "ip": ip,
                "asn": {
                    "asn": random.randint(1000,9000),
                    "as_org": "Example ASN Org"
                }
            }
        ]
    }


def generate(prefix, count, label):
    rows = []

    for i in range(count):
        if label == "malicious":
            domain = f"{prefix}{i}.com"
        else:
            domain = f"{prefix}{i}.com"

        rows.append(make_record(domain, label))

    return rows


malicious = generate("xqz", 15000, "malicious")
benign = generate("normal", 15000, "benign")


with open(OUT / "malware.json", "w") as f:
    json.dump(malicious, f)

with open(OUT / "benign_umbrella.json", "w") as f:
    json.dump(benign, f)


print("Created:")
print("malware.json:", len(malicious))
print("benign_umbrella.json:", len(benign))
