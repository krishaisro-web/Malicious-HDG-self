# Dataset Profile Report: Zenodo DomainRadar v2
*Generated at: 2026-10-05T05:02:57.626099+00:00*

## 1. Summary Counts & Schema Keys
| Dataset File | Record Count | DNS Null % | RDAP Null % | TLS Null % | IP Null % | No A/AAAA % |
|---|---|---|---|---|---|---|
| malware.json | 100,809 | 0.0% | 24.8% | 41.7% | 36.7% | 37.4% |
| benign_umbrella.json | 368,956 | 0.0% | 47.0% | 17.5% | 11.6% | 45.2% |
| benign_cesnet.json | 461,388 | 0.0% | 34.4% | 11.5% | 15.0% | 17.6% |

### Malware Family Key Conflict Inspection
- `malware` key present in: 0 records
- `malware_type` key present in: 100809 records

## 2. Temporal Feasibility & Split Validity
- **Time Split Valid**: `False`
  - *Reason*: Benign has 3 distinct sourced_on months (< 4).
- Distinct Months: Malware=6, Benign=3, Overlapping=1
- Sourced_on Range (Malware): 2023-07-12T00:10:41+00:00 to 2024-03-17T21:28:42+00:00

## 3. Domain Lexical & Subdomain Distributions
| Class | Subdomain % | e2LD % | Length (mean ± std) | Labels (mean ± std) |
|---|---|---|---|---|
| Malware | 24.9% | 75.1% | 17.97 ± 7.07 | 2.37 ± 0.69 |
| Umbrella | 74.5% | 25.5% | 21.61 ± 10.43 | 3.18 ± 1.05 |
| CESNET | 35.9% | 64.1% | 17.05 ± 7.62 | 2.87 ± 0.82 |

## 4. Key Counters & Schema Verification
- **ASN Candidate Keys Found**: {'asn': 1083964, 'as_org': 1083964, 'network_address': 1083964, 'prefix_len': 1083964}
- **Registrar Entity Keys Found**: {'handle': 133792, 'type': 134300, 'name': 266707, 'email': 8071, 'url': 22217, 'tel': 3629, 'whois_server': 3067}
- **DNS.NS Structure Types**: {'dict': 151586}
- **TLS Cert Fields**: ['common_name', 'organization', 'country', 'validity_start', 'validity_end', 'valid_len', 'extensions', 'extension_count', 'is_root']

## 5. Top Sources & Top Malware Families
- **Malware Sources**: ThreatFox (36407), Steven Black (27091), The Firebog (17131), Github (13421), MISP (3045), URLhaus (2647), Spam404 (1067)
- **Malware Families**: unknown (65154), QakBot (18044), Cobalt Strike (3543), IcedID (1973), Vidar (1599), Prometei (1440), Astaroth (1437), CryptBot (1361)

## 6. Runtime $defs.rdapEntity Schema
```json
{
  "title": "RDAP Entity",
  "description": "An entity fetched from RDAP. If WHOIS was used, only 'name' or 'email' will be present. Refer to the RDAP RFCs for more information on the possible fields and their meanings.",
  "type": "object",
  "properties": {
    "handle": {
      "type": "string"
    },
    "type": {
      "type": "string"
    },
    "name": {
      "type": "string"
    },
    "email": {
      "type": "string"
    }
  },
  "additionalProperties": {
    "anyOf": [
      {
        "type": "string"
      },
      {
        "type": "null"
      }
    ]
  }
}
```
