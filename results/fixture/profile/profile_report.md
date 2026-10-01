# [FIXTURE - meaningless] Dataset Profile Report: Zenodo DomainRadar v2
*Generated at: 2026-10-01T03:34:30.044143+00:00*

## 1. Summary Counts & Schema Keys
| Dataset File | Record Count | DNS Null % | RDAP Null % | TLS Null % | IP Null % | No A/AAAA % |
|---|---|---|---|---|---|---|
| malware.json | 3,000 | 5.5% | 7.6% | 35.5% | 5.5% | 13.2% |
| benign_umbrella.json | 1,500 | 5.1% | 9.0% | 34.3% | 5.1% | 13.0% |
| benign_cesnet.json | 1,500 | 5.1% | 7.6% | 35.7% | 5.1% | 12.2% |

### Malware Family Key Conflict Inspection
- `malware` key present in: 1493 records
- `malware_type` key present in: 1507 records

## 2. Temporal Feasibility & Split Validity
- **Time Split Valid**: `True`
- Distinct Months: Malware=16, Benign=16, Overlapping=16
- Sourced_on Range (Malware): 2023-04-01T12:00:00+00:00 to 2024-07-28T12:00:00+00:00

## 3. Domain Lexical & Subdomain Distributions
| Class | Subdomain % | e2LD % | Length (mean ± std) | Labels (mean ± std) |
|---|---|---|---|---|
| Malware | 27.1% | 72.9% | 17.28 ± 2.57 | 2.27 ± 0.44 |
| Umbrella | 27.5% | 72.5% | 17.35 ± 2.62 | 2.28 ± 0.45 |
| CESNET | 28.9% | 71.1% | 17.36 ± 2.57 | 2.29 ± 0.45 |

## 4. Key Counters & Schema Verification
- **ASN Candidate Keys Found**: {'number': 2902, 'organization': 8165, 'network': 8165, 'asn': 2602, 'autonomous_system_number': 2661}
- **Registrar Entity Keys Found**: {'name': 4138, 'handle': 4138, 'organization': 4138}
- **DNS.NS Structure Types**: {'dict': 4257}
- **TLS Cert Fields**: ['common_name', 'organization', 'country', 'validity_start', 'validity_end', 'valid_len', 'extensions', 'extension_count', 'is_root']

## 5. Top Sources & Top Malware Families
- **Malware Sources**: ThreatFox (616), URLhaus (615), abuse.ch (607), Firebog (586), MISP (576)
- **Malware Families**: emotet (462), formbook (441), unknown (439), qakbot (435), agenttesla (415), asyncrat (404), redline (404)

## 6. Runtime $defs.rdapEntity Schema
```json
{
  "type": "object",
  "description": "Entity structure inside RDAP object",
  "properties": {
    "name": {
      "type": "string"
    },
    "handle": {
      "type": "string"
    },
    "organization": {
      "type": "string"
    },
    "kind": {
      "type": "string"
    },
    "roles": {
      "type": "array",
      "items": {
        "type": "string"
      }
    }
  }
}
```
