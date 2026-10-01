# [FIXTURE - meaningless] Dataset Shortcut & Leakage Audit

## 1. Single-Feature ROC-AUCs
*(Domain and infrastructure features with AUC >= 0.90 indicate potential trivial shortcuts)*

| Feature | ROC-AUC | Status |
|---|---|---|
| `length` | 0.5050 | PASS: OK (< 0.90) |
| `num_labels` | 0.5000 | PASS: OK (< 0.90) |
| `subdomain_flag` | 0.5056 | PASS: OK (< 0.90) |
| `tld` | 0.5072 | PASS: OK (< 0.90) |
| `no_ip_nxdomain` | 0.5029 | PASS: OK (< 0.90) |
| `has_tls` | 0.5027 | PASS: OK (< 0.90) |
| `has_rdap` | 0.5033 | PASS: OK (< 0.90) |
| `source_only` | 1.0000 | Provenance Metadata (Excluded from Features) |

## 2. Class-by-Source Contingency Table
| Source Feed | Malware (1) | Benign (0) | Malware Ratio |
|---|---|---|---|
| CESNET | 0 | 1,461 | 0.0% |
| Firebog | 576 | 0 | 100.0% |
| MISP | 564 | 0 | 100.0% |
| ThreatFox | 602 | 0 | 100.0% |
| URLhaus | 598 | 0 | 100.0% |
| Umbrella | 0 | 1,459 | 0.0% |
| abuse.ch | 594 | 0 | 100.0% |

## 3. Audit Verdict
> **VERDICT: PASS.** No single domain or infrastructure feature trivially separates malware from benign (all domain AUCs < 0.90).
