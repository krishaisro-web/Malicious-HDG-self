# Operational Runbook: Executing Malicious-HDG on Real Zenodo Data

This runbook provides exact, copy-pasteable instructions for running the complete **Malicious-HDG** pipeline on the production CPU-only Linux server containing the genuine **Zenodo DomainRadar v2** dataset.

---

## 1. Server Environment & Hardware Specifications

| Component | Target Server Specification | Operational Guidance |
|:---|:---|:---|
| **OS** | Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9) | x86_64, standard POSIX shell |
| **CPU** | Multi-core enterprise server (e.g. 32 to 128 cores) | OpenMP PyTorch threading supported |
| **RAM** | ~245 GB physical RAM available | In-memory graph processing will utilize ~16–32 GB peak |
| **GPU** | None (CPU-only execution) | All models run on PyTorch CPU with OpenMP threading |
| **Python** | Python 3.12 (or 3.10+) | Pinned dependencies in `requirements-lock.txt` |
| **Disk** | $\ge 50$ GB free SSD space | Intermediate parquets and checkpoints require ~5–10 GB |

---

## 2. Session Management & Environment Setup

Because production training across 5 evaluation seeds takes 2–4 hours, **always execute inside a persistent `tmux` session** so SSH disconnects never interrupt the pipeline.

```bash
# 1. Start or attach to a persistent tmux session
tmux new -s hdg_production

# 2. Navigate to repository root
cd /path/to/Malicious-HDG

# 3. Create and activate a clean Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 4. Upgrade pip and install locked dependencies
pip install --upgrade pip
pip install -r requirements-lock.txt

# 5. Set PyTorch thread allocation (adjust based on core availability, e.g. 32 or 64)
export HDG_THREADS=32
```

---

## 3. Step-by-Step Pipeline Execution Sequence

### Step 0: Preflight System & Resource Check
**Script**: `src/scripts/00c_preflight.py`  
**Expected Duration**: ~15 seconds  
**Memory**: $< 1$ GB  

Verifies hardware, python dependencies, raw dataset presence, and executes a 1-epoch 10% stratified subsample throughput benchmark with runtime extrapolation.

```bash
python -m src.scripts.00c_preflight
```
> **VERIFICATION**: Must print `[PASS] System hardware & dataset checks passed!` Confirm all 4 raw files are detected and estimated runtime is acceptable.

---

### Step 1: Dataset Profiling & Schema Analysis
**Script**: `src/scripts/00_profile_dataset.py`  
**Expected Duration**: ~1–2 minutes  
**Memory**: $< 2$ GB (streaming JSON parser via `ijson`)  

Analyzes record counts, key candidate distribution for ASNs and Registrars, and verifies whether temporal rolling-origin evaluation is feasible.

```bash
python -m src.scripts.00_profile_dataset
```
> **ACTION**: Review `results/profile/profile_report.md` (compact, $\le 200$ lines). Verify that `time_split_valid.json` contains `"time_split_valid": true`.

---

### Step 2: Normalization, Parsing & Time-Matched Benign Cohort
**Script**: `src/scripts/01_parse.py`  
**Expected Duration**: ~3–5 minutes  
**Memory**: ~4–8 GB  

Normalizes domain names to e2LD, resolves candidate keys for ASN and Registrar, extracts dual certificate keys (CN and co-issue hash), extracts BGP prefixes, and samples a time-matched benign cohort.

```bash
python -m src.scripts.01_parse
```
> **GENERATED ARTIFACTS**:
> - `data_processed/domains.parquet` (Complete dataset)
> - `data_processed/domains_timematched.parquet` (Time-matched cohort)
> - `data_processed/parsing_summary.json`

---

### Step 3: Shortcut & Single-Feature Leakage Audit
**Script**: `src/scripts/00b_audit_shortcuts.py`  
**Expected Duration**: ~30 seconds  
**Memory**: ~2 GB  

Audits all domain features for label shortcut proxies. If any single domain feature achieves ROC-AUC $\ge 0.90$, it aborts with a diagnostic error to prevent spurious graph results.

```bash
python -m src.scripts.00b_audit_shortcuts
```
> **VERIFICATION**: Must report `Verdict: PASS`. If `subdomain_flag` achieves AUC $\ge 0.90$, confirm `include_has_subdomain: false` in `configs/default.yaml`.

---

### Step 4: Construct Unscaled Base Heterogeneous Graph
**Script**: `src/scripts/02_build_graph.py`  
**Expected Duration**: ~1–2 minutes (vectorized extraction)  
**Memory**: ~8–12 GB  

Constructs the unscaled base heterogeneous graph containing domain, ip, nameserver, registrar, asn, and certificate nodes. Standardizers are purposefully NOT fitted globally to prevent test-set data leakage.

```bash
python -m src.scripts.02_build_graph
```
> **GENERATED ARTIFACTS**:
> - `data_processed/heterodata.pt` (Unscaled base graph)
> - `data_processed/id_maps.json` (Topology entity index mappings)

---

### Step 5: Fast Smoke Validation on Real Data
**Script**: `src.scripts.run_all --smoke`  
**Expected Duration**: ~3–5 minutes  
**Memory**: ~8 GB  

Before committing to a multi-hour run, execute all remaining experimental stages in accelerated smoke mode on the real dataset to verify memory, convergence, and metric calculation end-to-end.

```bash
python -m src.scripts.run_all --smoke
```
> **VERIFICATION**: Ensure all stages complete with exit code 0. Confirm `results/smoke/REPORT.md` is generated.

---

### Step 6: Full Production Run

You have two equivalent options to execute the complete production evaluation:

#### Option A: Automated One-Command Script (Recommended)
Executes all stages sequentially with atomic run logging, timestamping, error trapping, and automatic results bundling:

```bash
bash scripts/run_real.sh
```

#### Option B: Step-by-Step CLI Execution
If preferred, you can invoke each experimental module individually. All scripts automatically check `results/runs/<name>.json` and skip completed stages if restarted (use `--force` to override):

```bash
# 1. Multi-Seed Baselines (TF-IDF, XGBoost tabular/lexical, MLP no-edges, Length, No-IP)
python -m src.scripts.03_run_baselines

# 2. HeteroGNN Multi-Split Training (SAGE, Attn across Random, Time, Group splits)
python -m src.scripts.04_train_gnn --variant both

# 3. Systematic Graph & Feature Ablations (35 variants)
python -m src.scripts.05_ablation

# 4. Structural Adversarial Attack & Relational GNNGuard Recovery
python -m src.scripts.06_attack_eval

# 5. Operational CPU Inference Latency Benchmark
python -m src.scripts.07_latency_eval --num-queries 500 --warmup 20

# 6. Dataset Provenance & Shortcut Generalization Checks
python -m src.scripts.08_provenance_checks

# 7. Continual Learning & Streaming Domain Updates
python -m src.scripts.09_streaming_eval

# 8. BIND Response Policy Zone (RPZ) Rule Emission
python -m src.scripts.10_emit_rpz --target-fpr 0.001

# 9. Final Comprehensive Report Compilation
python -m src.scripts.11_make_report
```

---

## 4. Packaging & Sending Results Back

Once the run completes, package the final evaluation report, tables, and run logs into an archive:

```bash
# Create compressed results bundle
tar -czvf results_bundle_$(date +%Y%m%d).tar.gz \
    results/REPORT.md \
    results/tables/ \
    results/runs/ \
    results/rpz/ \
    results/logs/
```

### Artifacts to Inspect and Return:
1. `results/REPORT.md`: The definitive master document summarizing objectives O1–O5, baseline comparisons, temporal generalization, adversarial recovery, latency, and RPZ blocking rates.
2. `results/tables/*.md` and `*.json`: All raw multi-seed tabular evaluations.
3. `results/rpz/malicious_domains.rpz`: Production-ready DNS Response Policy Zone rule file.
4. `results/logs/*.log`: Execution console logs with timing and system resource statistics.
