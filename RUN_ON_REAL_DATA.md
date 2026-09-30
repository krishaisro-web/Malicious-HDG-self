# Protocol for Executing Malicious-HDG on Real Data

This document provides the exact, step-by-step execution protocol for running the pipeline on the target computer containing the genuine **Zenodo DomainRadar v2** dataset.

All scripts must be executed from the **repository root directory**.

---

## Prerequisites & Environment Setup

Verify the Python environment and install the pinned dependencies:

```bash
# Verify Python version (Python 3.10 - 3.14 supported)
python --version

# Install pinned dependencies
pip install -r requirements.txt
```

---

## Step 1: Place Dataset Files in `data_raw/zenodo/`

Place the four raw dataset files into `data_raw/zenodo/`:
```
data_raw/
  └── zenodo/
        ├── malware.json            (~100,809 records)
        ├── benign_umbrella.json     (~368,956 records)
        ├── benign_cesnet.json       (~461,338 records)
        └── data_schema.json         (JSON schema definitions)
```

Confirm that all four files are present:
```bash
python -c "from pathlib import Path; files = ['malware.json', 'benign_umbrella.json', 'benign_cesnet.json', 'data_schema.json']; p = Path('data_raw/zenodo'); print({f: (p/f).exists() for f in files})"
```

---

## Step 2: Profile Dataset & Verify Schema

Run the streaming profiler. This step does **not** load full files into RAM; it streams records using `ijson` and produces structural presence rates, key counters, date distributions, and temporal feasibility checks.

```bash
python -m src.scripts.00_profile_dataset
```

### Artifacts to Review:
1. `results/profile/profile_report.json`
2. `results/profile/profile_report.md` (compact summary, <= 200 lines)
3. `results/profile/time_split_valid.json`

> **ACTION REQUIRED BEFORE PROCEEDING**:
> Send `results/profile/profile_report.md` back to the development team for inspection.
> Verify:
> - `time_split_valid` is `true`.
> - Key names observed for ASN and Registrar are populated.
> - Presence rates of `malware` vs `malware_type` keys.

---

## Step 3: Parse Dataset & Run Shortcut / Leakage Audit

Run data normalization, candidate key resolution, cross-class deduplication, and time-matched benign sampling:

```bash
# 1. Parse and extract domain infrastructure records
python -m src.scripts.01_parse

# 2. Run shortcut and single-feature AUC audit
python -m src.scripts.00b_audit_shortcuts
```

### Artifacts to Review:
1. `data_processed/domains.parquet`
2. `data_processed/parsing_summary.json`
3. `results/profile/audit_shortcuts.md`

> **ACTION REQUIRED BEFORE PROCEEDING**:
> Send `results/profile/audit_shortcuts.md` back to the development team.
> Verify:
> - **Verdict**: `PASS`. Confirm no single domain feature has ROC-AUC $\ge 0.90$.
> - If `subdomain_flag` has AUC $\ge 0.90$, verify that `include_has_subdomain: false` remains disabled in `configs/default.yaml`.

---

## Step 4: Construct Heterogeneous Graph

Build the PyG `HeteroData` graph structure and compute leak-free standardization statistics:

```bash
python -m src.scripts.02_build_graph
```

### Generated Artifacts:
- `data_processed/heterodata.pt` (PyG HeteroData with domain, ip, nameserver, registrar, asn, certificate nodes)
- `data_processed/id_maps.json` (Entity index mappings)
- `data_processed/scaler_stats.json` (Train-only standardization parameters)

---

## Step 5: Execute Experimental Suite

Once steps 1–4 are verified, execute the experimental evaluation suite:

### 5.1 Run Baselines on Identical Splits
Evaluates TF-IDF on e2LD, XGBoost Tabular, and XGBoost Tabular + Lexical across all 5 evaluation seeds:
```bash
python -m src.scripts.03_run_baselines
```
Results saved to `results/tables/baselines_results.md` and `.json`.

### 5.2 Train and Evaluate HeteroGNN Models
Trains both **SAGE** and **Attn** (SHetGCN-inspired) architectures across all 5 seeds on Random, Time, and Group splits. Computes exact McNemar tests and paired bootstrap AUC differences:
```bash
# Random Stratified Split:
python -m src.scripts.04_train_gnn --variant both --split-type random

# Group Split (Bipartite Connected Components & Primary ASN):
python -m src.scripts.04_train_gnn --variant both --split-type group

# Temporal Split (Forward Time Generalization):
python -m src.scripts.04_train_gnn --variant both --split-type time
```
Results saved to `results/tables/gnn_evaluation_*.md` and `.json`.

### 5.3 Run Relation Ablations
Evaluates graph contributions when removing certificates, nameservers, registrars, or ASNs:
```bash
python -m src.scripts.05_ablation
```
Results saved to `results/tables/ablation_results.md` and `.json`.

### 5.4 Evaluate Adversarial Structural Robustness
Simulates adversarial edge injection across budgets $k \in \{1, 2, 5\}$:
```bash
python -m src.scripts.06_attack_eval
```
Results saved to `results/tables/attack_results.md` and `.json`.

### 5.5 Measure Operational CPU Latency
Benchmarks single-query inference latency across feature lookup, forward pass, and end-to-end execution:
```bash
python -m src.scripts.07_latency_eval --num-queries 100 --warmup 10
```
Results saved to `results/tables/latency_results.md` and `.json`.

---

## One-Command Alternative: Complete Automated Pipeline

To run all steps sequentially in a single automated batch:
```bash
python -m src.scripts.run_all
```
