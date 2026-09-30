# Problem Scope: Heterogeneous Graph-Based Malware Domain Detection

## 1. Problem Formulation & Task Definition

The objective of **Malicious-HDG** is the automated binary classification of internet domain names into:
- **Malicious Domain (Label 1)**: Domains observed distributing malware, hosting command-and-control (C2) servers, or participating in malicious botnet campaigns.
- **Benign Domain (Label 0)**: Legitimate, benign domains active on the public internet.

The classification is performed using a **Heterogeneous Information Network (HIN)** constructed from multi-modal infrastructure dependencies, combined with domain-level technical and lexical features.

> **Important Clarification on Dataset Provenance**:
> Prior legacy drafts of this repository referred to DGArchive and wordlist-based Domain Generation Algorithms (DGA). Those references were incorrect and unsupported. The target dataset is **Zenodo DomainRadar v2**, which aggregates real-world malware feeds (ThreatFox, URLhaus, Firebog, MISP), **not** algorithmic DGA archives. All modeling and evaluation in this repository strictly reflect malware infrastructure detection.

---

## 2. Target Dataset: Zenodo DomainRadar v2

The pipeline is designed for the published **DomainRadar v2** dataset:
- **Citation**: Hranický, R. et al., *"DomainRadar: A Large-Scale Multi-Modal Dataset of Malicious and Benign Domains"*, *Data in Brief*, 2025. DOI: [10.1016/j.dib.2025.112062](https://doi.org/10.1016/j.dib.2025.112062).
- **Zenodo Repository**: DOI: [10.5281/zenodo.14332167](https://doi.org/10.5281/zenodo.14332167). License: **CC-BY 4.0**.
- **Collection Window**: March 2023 – July 2024.
- **Primary Source Files**:
  1. `malware.json` (100,809 domains): Multi-source malware feeds from ThreatFox, URLhaus, Firebog, MISP, and abuse.ch.
  2. `benign_umbrella.json` (368,956 domains): Cisco Umbrella Top 1M popular benign domains.
  3. `benign_cesnet.json` (461,338 domains): Passive DNS and active scan observations from CESNET research network.
  4. `data_schema.json`: JSON schema specifications and entity definitions.

---

## 3. Graph Schema & Node/Edge Architecture

The infrastructure graph is modeled as a PyTorch Geometric `HeteroData` structure:

### Node Types
| Node Type | Description | Feature Representation |
|---|---|---|
| `domain` | The classified entity (e2LD or FQDN) | Standardized 17D vector (technical DNS/RDAP/TLS flags + lexical stats) |
| `ip` | IP addresses resolved via DNS A/AAAA or NS related records | Log degree in experiment graph `log(1 + deg)` |
| `nameserver` | Authoritative DNS nameservers | Log degree in experiment graph `log(1 + deg)` |
| `registrar` | Domain registrar entity | Log degree in experiment graph `log(1 + deg)` |
| `asn` | Autonomous System Number of hosting IPs | Log degree in experiment graph `log(1 + deg)` |
| `certificate` | Leaf TLS certificate (first non-root certificate) | Log degree in experiment graph `log(1 + deg)` |

### Relation Types (Edges)
All relations are bidirectional (each canonical relation includes its reverse relation):
1. `('domain', 'resolves_to', 'ip')` & `('ip', 'rev_resolves_to', 'domain')`: **Strictly filtered to records with `from_record in {A, AAAA}`**. NS and MX IPs are excluded from direct resolution edges.
2. `('nameserver', 'ns_ip', 'ip')` & `('ip', 'rev_ns_ip', 'nameserver')`: Glue records connecting nameserver hostnames to their respective IP addresses.
3. `('domain', 'uses_ns', 'nameserver')` & `('nameserver', 'rev_uses_ns', 'domain')`: Authoritative nameserver delegation from `dns.NS` and `rdap.nameservers`.
4. `('domain', 'registered_by', 'registrar')` & `('registrar', 'rev_registered_by', 'domain')`: Domain sponsorship extracted from `rdap.entities.registrar[]`.
5. `('ip', 'belongs_to_asn', 'asn')` & `('asn', 'rev_belongs_to_asn', 'ip')`: Network topology mapping IP addresses to BGP routing origins.
6. `('domain', 'uses_cert', 'certificate')` & `('certificate', 'rev_uses_cert', 'domain')`: Observed TLS binding between a domain and its leaf X.509 certificate.

---

## 4. Cumulative Temporal Formulation vs. Legacy Snapshots

The legacy implementation split data into isolated weekly graph snapshots. However, empirical analysis shows that domains in single-collection datasets contribute rich relational information in only one observation window, causing multi-step recurrent aggregators (GRUs) to process predominantly empty timesteps.

**The Cumulative Formulation**:
- Rather than disconnected weekly islands, we construct **one cumulative graph per experiment** containing domains with $t \le t_{\text{end}}$.
- **No Future Edges**: An edge is included if and only if its originating domain was observed at or before $t_{\text{end}}$.
- For random and group splits, the full cumulative graph is utilized with strict train-only feature standardization.

---

## 5. Leak-Free Evaluation Protocol

To prevent metric inflation and data leakage:
1. **Train-Only Standardization**: All feature scalers (mean, standard deviation) are fit exclusively on the training domains/nodes.
2. **Disjoint Multi-Split Evaluation**:
   - **Random Stratified Split**: Optimistic upper bound.
   - **Temporal Split**: Evaluates forward generalization ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$), conditional on overlapping longitudinal coverage.
   - **Group Split (Bipartite Connected Components)**: Prunes infrastructure hubs (degree > 500 or top 0.1%) and splits by connected components, ensuring no component spans train and test. If giant component > 50%, groups by primary ASN.
3. **Threshold Calibration**: Decision thresholds are tuned strictly on validation splits (optimizing F1), then fixed for test evaluation.
4. **Statistical Rigor**:
   - All models evaluated across 5 random seeds (reporting mean ± std, never "best seed").
   - 95% empirical bootstrap confidence intervals (1,000 resamples).
   - Exact McNemar tests (`min(b,c)` statistic) and paired bootstrap AUC differences for paired model comparisons.
