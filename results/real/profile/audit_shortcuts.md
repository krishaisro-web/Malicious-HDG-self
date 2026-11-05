# Dataset Shortcut & Leakage Audit

## 1. Single-Feature ROC-AUCs
*(Domain and infrastructure features with AUC >= 0.90 indicate potential trivial shortcuts)*

| Feature | ROC-AUC | Status |
|---|---|---|
| `length` | 0.6211 | PASS: OK (< 0.90) |
| `num_labels` | 0.6342 | PASS: OK (< 0.90) |
| `subdomain_flag` | 0.5091 | PASS: OK (< 0.90) |
| `tld` | 0.8737 | PASS: OK (< 0.90) |
| `no_ip_nxdomain` | 0.6267 | PASS: OK (< 0.90) |
| `has_tls` | 0.6492 | PASS: OK (< 0.90) |
| `has_rdap` | 0.5168 | PASS: OK (< 0.90) |
| `source_only` | 1.0000 | Provenance Metadata (Excluded from Features) |

## 2. Class-by-Source Contingency Table
| Source Feed | Malware (1) | Benign (0) | Malware Ratio |
|---|---|---|---|
| ../../virus-total/finished_collections/benign_groundtruth2312.txt | 0 | 128,753 | 0.0% |
| Github | 12,186 | 0 | 100.0% |
| MISP | 2,418 | 0 | 100.0% |
| Spam404 | 989 | 0 | 100.0% |
| Steven Black | 16,913 | 0 | 100.0% |
| The Firebog | 15,691 | 0 | 100.0% |
| ThreatFox | 32,806 | 0 | 100.0% |
| URLhaus | 2,351 | 0 | 100.0% |
| data/alexa.csv | 0 | 19,415 | 0.0% |
| data/cisco_umbrella_year_filtered_domains.csv | 0 | 18,540 | 0.0% |

## 3. Audit Verdict
> **VERDICT: PASS.** No single domain or infrastructure feature trivially separates malware from benign (all domain AUCs < 0.90).
