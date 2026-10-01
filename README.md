# Malicious-HDG: Heterogeneous Graph-Based Malware Domain Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.5%2B-green.svg)](https://pyg.org/)
[![License: CC-BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**Malicious-HDG** is a leak-free, empirically grounded machine learning framework for detecting malicious internet domains using **Heterogeneous Graph Neural Networks (HIN)**. It constructs multi-modal infrastructure graphs linking domain names to resolved IP addresses, authoritative nameservers, domain registrars, Autonomous System Numbers (ASNs), and leaf X.509 TLS certificates.

---

## 1. Problem Definition & Dataset Provenance

### Task Definition
Binary classification of domain names into **Malware (Label 1)** vs **Benign (Label 0)** based on their structural graph topology and technical domain attributes.

> **Dataset Scope**:
> This repository strictly targets malware infrastructure detection using feeds from **ThreatFox, URLhaus, Firebog, MISP, and abuse.ch**. It does **not** model DGArchive or algorithmic DGA wordlists.

### Dataset Citation
The target dataset is **Zenodo DomainRadar v2**:
- **Dataset Reference**: Hranický, R., et al., *"DomainRadar: A Large-Scale Multi-Modal Dataset of Malicious and Benign Domains"*, *Data in Brief*, 2025. DOI: [10.1016/j.dib.2025.112062](https://doi.org/10.1016/j.dib.2025.112062).
- **Zenodo DOI**: [10.5281/zenodo.14332167](https://doi.org/10.5281/zenodo.14332167).
- **License**: Creative Commons Attribution 4.0 International ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Temporal Collection**: March 2023 – July 2024.
- **Dataset Scale**:
  - `malware.json`: ~100,809 malicious domains.
  - `benign_umbrella.json`: ~368,956 benign domains (Cisco Umbrella Top 1M).
  - `benign_cesnet.json`: ~461,338 benign domains (CESNET research network scans).
  - `data_schema.json`: Formal JSON schema definitions.

---

## 2. Graph Schema & Architectures

### Node & Edge Schema
The infrastructure graph models 6 node types and 12 directed relation types:
- **Node Types**: `domain`, `ip`, `nameserver`, `registrar`, `asn`, `certificate`.
- **Relations**:
  - `(domain, resolves_to, ip)` & `(ip, rev_resolves_to, domain)`: DNS A/AAAA resolution records.
  - `(nameserver, ns_ip, ip)` & `(ip, rev_ns_ip, nameserver)`: Nameserver glue IP mappings.
  - `(domain, uses_ns, nameserver)` & `(nameserver, rev_uses_ns, domain)`: Authoritative nameserver delegations.
  - `(domain, registered_by, registrar)` & `(registrar, rev_registered_by, domain)`: Domain registrar entities.
  - `(ip, belongs_to_asn, asn)` & `(asn, rev_belongs_to_asn, ip)`: BGP autonomous system routing origins.
  - `(domain, uses_cert, certificate)` & `(certificate, rev_uses_cert, domain)`: Leaf TLS certificates (`is_root == False`).

### Model Variants
1. **`HeteroGNN (sage)`**: Relational GraphSAGE with linear input projection, mean neighborhood aggregation, and residual connections.
2. **`HeteroGNN (attn)`**: Relational Heterogeneous Attention using `HGTConv`, inspired by SHetGCN.
3. **`HeteroGNN (sage_guard)`**: Relational GNNGuard variant with cosine edge pruning, row-normalization, learnable layer memory, and weighted aggregation for adversarial defense.

---

## 3. PhD Target Operational Objectives (O1–O5)

| Metric | Target Criterion | Operational Significance |
|:---|:---|:---|
| **O1: Low-FPR Operation** | Operational FPR $\le 0.1\%$ with non-trivial TPR | Prevents DNS resolver denial-of-service on benign traffic. |
| **O2: Temporal Generalization** | F1 degradation $\le 0.10$ on $T \to T+1$ | Evaluates forward robustness against infrastructure turnover. |
| **O3: AUROC Superiority** | Significant gain over tabular XGBoost ($p < 0.05$) | Demonstrates structural graph value over isolated features. |
| **O4: CPU Latency Budget** | Per-query CPU inference $< 50$ ms | Enables inline real-time DNS resolver inspection without GPUs. |
| **O5: Adversarial Robustness** | $\ge 70\%$ recovery score under structural attack | Defends against coordinated edge-injection evasion attacks. |

---

## 4. Methodological Guarantees (Leak-Free ML)

- **Unscaled Base Graph Formulation**: The base heterogeneous graph (`heterodata.pt`) is cached without standardizers to eliminate data leakage.
- **Split-Specific Standardization**: `apply_split_scaling` computes $\mu, \sigma$ strictly on training domains and train-incident infrastructure per split.
- **Multi-Split Paradigm**: Evaluates Random Stratified (optimistic bound), Rolling-Origin Temporal ($T \to T+1$), and Stratified Group Splits (dual-pool bipartite connected components and ASN grouping).
- **Validation-Calibrated Thresholds**: Decision thresholds are calibrated on validation negatives to guarantee operational FPR $\le 0.1\%$.
- **Statistical Rigor**: 5 evaluation seeds (mean ± std), 95% bootstrap confidence intervals, exact McNemar tests (`min(b,c)` statistic), and paired bootstrap AUC difference tests.
- **Continual Learning**: Streaming prototype comparing full retrain from scratch, naive fine-tuning, and reservoir replay buffer ($M=1000$).
- **BIND RPZ Rule Emission**: Calibrates 0.1% FPR operational threshold and generates production-ready BIND Response Policy Zone rules (`<domain> CNAME .`).

---

## 5. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/krishaisro-web/Malicious-HDG.git
cd Malicious-HDG

# Install pinned dependencies
pip install -r requirements-lock.txt
```

### Local Validation via Synthetic Fixture
The repository includes self-contained synthetic session fixtures. Run full smoke test suite on CPU:
```bash
# 1. Execute pytest suite (32 comprehensive unit and integration tests)
pytest tests/ -v

# 2. Run full automated pipeline in fixture mode (completes in ~3 minutes on CPU)
python -m src.scripts.run_all --fixture --smoke
```

### Execution on Production CPU Linux Server
For instructions on deploying and evaluating on the server containing the real Zenodo dataset (~245 GB RAM, Python 3.12, CPU-only), see **[`docs/RUN_ON_REAL_DATA.md`](docs/RUN_ON_REAL_DATA.md)** and **[`pipeline/run_real.sh`](pipeline/run_real.sh)**.
For documented dataset assumptions and verification fields, see **[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md)**.
For a complete record of architectural refactoring, see **[`CHANGELOG.md`](CHANGELOG.md)**.

---

## 6. Repository Structure

```
Malicious-HDG/
├── README.md  LICENSE  CHANGELOG.md  .gitignore
├── requirements.txt  requirements-lock.txt
│
├── configs/                  # default.yaml pipeline configurations
├── src/hdg/                  # The ONLY importable package
│   ├── config.py             # Config loader, thread control & resumability
│   ├── metrics.py            # Prevalence-adjusted precision, McNemar, bootstrap
│   ├── data/                 # parse.py, fixture.py, profiler.py, audit.py, splits.py, graph.py
│   ├── models/               # hetero_gnn.py (SAGE, Attn, GNNGuard)
│   ├── training/             # train.py, baselines.py
│   └── eval/                 # attack.py, latency.py, streaming.py
├── pipeline/                 # Numbered pipeline scripts 00 → 11 + run_all.py + run_real.sh
├── experiments/              # Standalone side experiments (chrmor_lexical)
├── tests/                    # Active pytest suite (31 tests passing)
├── tools/                    # Utility scripts (make_schema_fixture.py)
├── data/
│   ├── raw/zenodo/           # Real Zenodo data (gitignored)
│   ├── fixture/zenodo/       # Small synthetic fixture data (committed)
│   └── processed/            # Processed parquets & graphs ({fixture, real}/)
├── artifacts/
│   └── checkpoints/          # Model checkpoints ({fixture, real}/)
├── results/
│   ├── fixture/              # Smoke & test run outputs (tables, runs, logs, rpz, profile)
│   ├── real/                 # Real data outputs
│   └── chrmor/               # Auxiliary lexical study outputs
├── docs/
│   ├── ASSUMPTIONS.md        # Real data assumptions & verification fields
│   ├── RUN_ON_REAL_DATA.md   # Exact ordered guide for Linux CPU server
│   ├── latency.md            # Inference latency specifications
│   ├── problem_scope.md      # Domain problem definition & scope
│   └── reports/              # Research reports & PDF summaries
└── archive/                  # FROZEN legacy v1 code, checkpoints, and tests (not tested/imported)
    └── legacy_v1/
        ├── README.md         # Archival documentation
        ├── scripts/          # Legacy v1 scripts
        ├── models/           # Legacy exploratory models
        ├── checkpoints/      # Legacy v1 weights
        └── results/          # Legacy v1 experiment outputs
```
