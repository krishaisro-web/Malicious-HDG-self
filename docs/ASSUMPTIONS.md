# Dataset Assumptions & Verification Protocol

In accordance with strict empirical ML methodology, no dataset statistics, performance numbers, or source lists are assumed or hardcoded. Every operational assumption regarding the target dataset (**Zenodo DomainRadar v2**) is documented below with its exact verifying field in the profiling report.

When an assumption is violated on real data, the codebase is designed to **fail loudly** rather than silently corrupting downstream models.

---

## 1. Summary of Explicit Assumptions

| # | Domain / Component | Assumption Description | Verifying Field in Profile Report (`profile_report.json` / `.md`) | Code Behavior on Violation |
|---|---|---|---|---|
| **1** | **Malware Family Key Conflict** | In `malware.json`, the malware family key is either named `malware` or `malware_type` (never null, though may be `"unknown"`). | `malware_key_count`, `malware_type_key_count` under `classes.malware` | Parser checks both `r.get("malware_type") or r.get("malware")`. Counts of both keys are logged in `profile_report.json`. |
| **2** | **ASN Key Candidate Resolution** | Inside `ip_data[].asn`, the Autonomous System Number is represented by one of `["asn", "autonomous_system_number", "number"]`. | `ip_asn_observed_keys` in `profile_report.json` | Parser attempts candidates in sequence. If **zero** ASNs are resolved across the entire dataset, code **fails loudly** (`ValueError: Zero ASNs were resolved using candidate keys...`). |
| **3** | **Registrar Entity Key Resolution** | Inside `rdap.entities.registrar[]`, the registrar identifier is represented by one of `["name", "handle", "organization"]`. | `rdap_registrar_entity_observed_keys`, and runtime `$defs.rdapEntity` from `data_schema.json` | Scans candidate keys in order and logs key usage count in `parsing_summary.json`. |
| **4** | **Timestamp Formats & Fallback** | Timestamps in `sourced_on` and `evaluated_on` are formatted as Mongo ISO strings `{"$date": "..."}`, numberLong timestamps `{"$date": {"$numberLong": "..."}}`, or plain ISO strings. | `distinct_sourced_on_dates_count`, `distinct_evaluated_on_dates_count` | `parse_mongo_date` converts all representations. Records lacking both timestamps are **excluded** and counted in `raw_counts.excluded_no_time` (never filled with minimum date). |
| **5** | **Temporal Split Validity** | Both malware and benign domains possess at least $N$ distinct observation months (default 4) of `sourced_on`, and their monthly ranges overlap. | `time_split_valid` in `time_split_valid.json` | If `time_split_valid` is `False`, temporal experiments are **disabled**, and attempts to force temporal splits raise `ValueError`. |
| **6** | **Graph Hub Pruning & Connected Components** | The domain-infrastructure bipartite graph contains high-degree infrastructure hubs. Pruning nodes with degree > 500 (or top 0.1%) breaks global component collapse. | `giant_component_fraction` in `splits.metadata` | If `giant_component_fraction > 0.50` (giant component contains > 50% of domains), the code emits a warning and **automatically provides group split by primary ASN**. |
| **7** | **Absence of Single-Feature Shortcuts** | Malware and benign domains are not trivially separable by single domain/infrastructure metadata features (e.g. length, label count, NXDOMAIN, TLS presence). | `single_feature_aucs`, `features_with_auc_ge_0_90` in `audit_shortcuts.json` | Any domain feature with ROC-AUC $\ge 0.90$ triggers a `[WARNING]` verdict. If `subdomain_flag` has AUC $\ge 0.90$, `include_has_subdomain` remains `false`. |
| **8** | **Certificate Identification (No Fingerprint/Serial)** | Zenodo TLS records lack X.509 serial numbers or SHA-256 fingerprints. The leaf cert must be selected using the `is_root == False` flag. | `tls_cert_field_presence` in `profile_report.json` | First cert with `is_root == False` is hashed over `(common_name, organization, country, validity_start, validity_end)`. Collisions between identical metadata certificates are documented as expected. |
| **9** | **2-Hop Ego-Subgraph Mathematical Equivalence** | For a 2-layer HeteroGNN, the representation of domain $u$ depends strictly on its 2-hop structural neighborhood $\mathcal{N}^2(u)$. | Verified in `tests/test_latency.py` | Extracts localized 2-hop ego-subgraph for inference on CPU, achieving $< 50$ ms query latency mathematically identical to full-graph slicing within $10^{-5}$. |
| **10** | **Leak-Free Split Standardization** | Fitting standardizers globally across all domains leaks test distribution statistics into training features. | Verified in `tests/test_graph_and_splits.py` | The base graph is cached unscaled (`heterodata.pt`). `apply_split_scaling` fits mean and std strictly on training domains and train-incident infrastructure per split. |
| **11** | **Stratified Group Prevalence Balancing** | Random component assignment causes severe class imbalance when clusters are predominantly pure. | `max_discrepancy` in `assign_groups_stratified` | Employs dual-pool distribution of positive-majority and negative-majority clusters to guarantee train/val/test prevalence within 5 percentage points of global. |
| **12** | **Adversarial Threat Model Target Constraints** | In PPT threat model, attackers cannot observe future test infrastructure and must connect to known legitimate infrastructure. | Verified in `tests/test_attack.py` | Candidate target infrastructure for edge injection is strictly ranked using training benign domains only. |
| **13** | **Operational Low-FPR Decision Calibration** | Real-world DNS resolvers require FPR $\le 0.1\%$ to prevent service disruption of legitimate traffic. | Verified in `pipeline/10_emit_rpz.py` | Calibrates operational threshold strictly on validation negatives at the 99.9th percentile ($1 - \text{FPR}_{\text{target}}$). |
| **14** | **Streaming JSON Parser Error Tolerance** | Malformed lines in raw multi-gigabyte JSON dumps should not abort ingestion unless systemic. | `malformed_count` in `parsing_summary.json` | Parser tolerates malformed records up to 1.0% of total lines. If malformed records exceed 1.0%, code aborts loudly. |

---

## 2. Verification Protocol for Target Machine

Prior to training models on the target machine, the operator must execute the verification protocol:

1. **Step 1: Execute Preflight System Verification**:
   ```bash
   python -m pipeline.00c_preflight
   ```
2. **Step 2: Execute Dataset Profiling**:
   ```bash
   python -m pipeline.00_profile_dataset
   ```
3. **Step 3: Inspect Output Reports**:
   - Check `results/real/profile/profile_report.md` (<= 200 lines).
   - Verify `time_split_valid.json` (`"time_split_valid": true`).
   - Check whether `malware` or `malware_type` was detected in raw records.
   - Verify that ASN keys and Registrar entity keys are non-empty.
4. **Step 4: Execute Shortcut Audit**:
   ```bash
   python -m pipeline.01_parse
   python -m pipeline.00b_audit_shortcuts
   ```
   - Check `results/real/profile/audit_shortcuts.md`.
   - Confirm that all domain features achieve AUC < 0.90 (Verdict: PASS).
