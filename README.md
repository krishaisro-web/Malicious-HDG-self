# Malicious-HDG: Heterogeneous Graph-Based Malware Domain Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5%2B-green.svg)](https://pyg.org/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Malicious-HDG** is a leak-free machine learning framework for detecting malicious internet domains using **Heterogeneous Graph Neural Networks (HIN)**. It constructs cumulative multi-modal infrastructure graphs linking domain names to resolved IP addresses, authoritative nameservers, domain registrars, Autonomous System Numbers (ASNs), and leaf X.509 TLS certificates.

---

## 1. Problem Definition & Dataset Provenance

### Task Definition
Binary classification of domain names into **Malware (Label 1)** vs **Benign (Label 0)** based on their graph infrastructure topology and technical domain attributes.

> **Provenance Correction**:
> This repository strictly targets malware infrastructure detection using feeds from **ThreatFox, URLhaus, Firebog, MISP, and abuse.ch**. It does **not** model DGArchive or algorithmic DGA wordlists.

### Dataset Citation
The target dataset is **Zenodo DomainRadar v2**:
- **Dataset Reference**: Hranický, R., et al., *"DomainRadar: A Large-Scale Multi-Modal Dataset of Malicious and Benign Domains"*, *Data in Brief*, 2025. DOI: [10.1016/j.dib.2025.112062](https://doi.org/10.1016/j.dib.2025.112062).
- **Zenodo DOI**: [10.5281/zenodo.14332167](https://doi.org/10.5281/zenodo.14332167).
- **License**: Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Temporal Collection**: March 2023 – July 2024.
- **Dataset Scale**:
  - `malware.json`: 100,809 malicious domains.
  - `benign_umbrella.json`: 368,956 benign domains (Cisco Umbrella Top 1M).
  - `benign_cesnet.json`: 461,338 benign domains (CESNET research network scans).
  - `data_schema.json`: Formal JSON schema definitions.

---

## 2. Graph Schema & Architectures

### Node & Edge Schema
The infrastructure graph models 6 node types and 12 directed relation types:
- **Node Types**: `domain`, `ip`, `nameserver`, `registrar`, `asn`, `certificate`.
- **Relations**:
  - `(domain, resolves_to, ip)` & `(ip, rev_resolves_to, domain)`: Strictly from DNS A/AAAA records.
  - `(nameserver, ns_ip, ip)` & `(ip, rev_ns_ip, nameserver)`: Nameserver glue IP mappings.
  - `(domain, uses_ns, nameserver)` & `(nameserver, rev_uses_ns, domain)`: Authoritative nameserver delegations.
  - `(domain, registered_by, registrar)` & `(registrar, rev_registered_by, domain)`: Domain registrar entities.
  - `(ip, belongs_to_asn, asn)` & `(asn, rev_belongs_to_asn, ip)`: BGP autonomous system routing origins.
  - `(domain, uses_cert, certificate)` & `(certificate, rev_uses_cert, domain)`: Leaf TLS certificates (selected via `is_root == False`).

### Model Variants
1. **`HeteroGNN (sage)`**: Relational GraphSAGE using `HeteroConv` + `SAGEConv` with mean neighborhood aggregation.
2. **`HeteroGNN (attn)`**: Relational Heterogeneous Attention using `HGTConv`, inspired by SHetGCN.

---

## 3. Methodological Guarantees (Leak-Free ML)

- **Cumulative Graph Formulation**: Replaced isolated weekly snapshots with a single cumulative graph ($t \le t_{\text{end}}$), eliminating future-edge leakage.
- **Train-Only Standardization**: All feature standardizers ($\mu, \sigma$) are fit strictly on training splits and saved to `scaler_stats.json`.
- **Multi-Split Paradigm**: Evaluates Random Stratified (optimistic bound), Forward Temporal (train < val < test by $t$), and Group Splits (bipartite connected components with hub removal, fallback to primary ASN).
- **Validation-Tuned Decision Thresholds**: Decision thresholds are calibrated on validation F1 and applied to test.
- **Statistical Rigor**: 5 evaluation seeds (mean ± std), 95% bootstrap confidence intervals (1,000 resamples), exact McNemar tests (`min(b,c)` statistic), and paired bootstrap AUC differences.
- **Adversarial Robustness & Operational Latency**: Built-in structural edge-injection attack evaluation ($k \in \{1, 2, 5\}$) and CPU per-query latency benchmarking.

---

## 4. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/krishaisro-web/Malicious-HDG.git
cd Malicious-HDG

# Install pinned dependencies
pip install -r requirements.txt
```

### Local Validation via Synthetic Fixture
The repository includes a schema-faithful synthetic fixture generator. Run the full smoke test suite on CPU:
```bash
# 1. Run full automated pipeline in fixture mode (completes in ~2 minutes on CPU)
python -m src.scripts.run_all --fixture

# 2. Execute pytest suite (17 comprehensive tests)
pytest tests/ -v
```

### Execution on Target Machine with Real Data
For instructions on deploying and evaluating on the machine containing the real Zenodo dataset, see **[`RUN_ON_REAL_DATA.md`](RUN_ON_REAL_DATA.md)**.
For documented dataset assumptions and verification fields, see **[`ASSUMPTIONS.md`](ASSUMPTIONS.md)**.
For a complete record of architectural refactoring, see **[`CHANGELOG.md`](CHANGELOG.md)**.

---

## 5. Repository Structure

```
Malicious-HDG/
├── configs/
│   └── default.yaml               # Unified experiment configuration
├── docs/
│   ├── problem_scope.md           # Formal task definition & dataset scope
│   └── latency.md                 # CPU inference latency protocol
├── models/
│   ├── hetero_gnn.py              # HeteroGNN (SAGE and Attn variants)
│   └── full_model.py              # Legacy baseline model
├── src/
│   ├── hdg/                       # Core Malicious-HDG package
│   │   ├── config.py              # Config & path resolution safeguards
│   │   ├── fixture.py             # Schema-faithful synthetic fixture
│   │   ├── profiler.py            # Dataset streaming profiler & time-split check
│   │   ├── parse.py               # Normalizer, deduplicator & candidate resolver
│   │   ├── audit.py               # Shortcut detector & single-feature AUC audit
│   │   ├── graph.py               # Heterogeneous cumulative graph builder
│   │   ├── splits.py              # Random, time, and group split generators
│   │   ├── train.py               # Unified training & evaluation loop
│   │   ├── baselines.py           # TF-IDF, XGBoost tabular/lexical baselines
│   │   ├── metrics.py             # Bootstrap CIs, exact McNemar & paired AUC
│   │   ├── attack.py              # Structural adversarial attack evaluator
│   │   └── latency.py             # CPU query latency benchmark
│   └── scripts/                   # Thin CLI execution scripts
│       ├── 00_profile_dataset.py  # Profile raw JSON files
│       ├── 00b_audit_shortcuts.py # Audit shortcuts on parsed data
│       ├── 01_parse.py            # Parse & extract Parquet
│       ├── 02_build_graph.py      # Build HeteroData graph
│       ├── 03_run_baselines.py    # Run TF-IDF and XGBoost
│       ├── 04_train_gnn.py        # Train HeteroGNN across seeds
│       ├── 05_ablation.py         # Run relation ablations
│       ├── 06_attack_eval.py      # Adversarial edge injection
│       ├── 07_latency_eval.py     # CPU latency measurement
│       ├── nodes_edges_count.py   # Fixed node/edge counting tool
│       └── run_all.py             # End-to-end pipeline runner
├── tests/                         # Pytest suite (17 tests)
├── ASSUMPTIONS.md                 # Real data assumptions & verification fields
├── CHANGELOG.md                   # Detailed record of fixes and changes
├── RUN_ON_REAL_DATA.md            # Exact ordered guide for target machine
└── requirements.txt               # Pinned dependencies
```
