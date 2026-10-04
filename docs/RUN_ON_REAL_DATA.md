# Operational Runbook: Executing Malicious-HDG on Real Zenodo Data

This runbook provides exact, copy-pasteable instructions for running the complete **Malicious-HDG** pipeline on the production CPU-only Linux server containing the genuine **Zenodo DomainRadar v2** dataset.

---

## 1. Supported Environments & Hardware Profiles

Malicious-HDG supports two target CPU deployment profiles:

### Profile A: Enterprise Multi-Core Server
| Component | Enterprise Specification | Operational Guidance |
|:---|:---|:---|
| **OS** | Linux (Ubuntu 22.04 LTS / Debian 12 / RHEL 9) | x86_64, standard POSIX shell |
| **CPU** | Multi-core enterprise server (e.g. 16 to 128 cores) | OpenMP PyTorch threading supported |
| **RAM** | $\ge 32$ GB physical RAM | In-memory graph processing utilizes ~8–16 GB peak |
| **GPU** | None (CPU-only execution) | PyTorch CPU with OpenMP threading |
| **Config** | `configs/default.yaml` | 5 seeds, 200 epochs, 100k domains |

### Profile B: HP Thin Client (ISRO Lab / Workstation Environment)
> **Note**: For complete details, thermal tuning, and Reddit-tested sysadmin recommendations, see **[`docs/HP_THIN_CLIENT_GUIDE.md`](HP_THIN_CLIENT_GUIDE.md)**.

| Component | HP Thin Client Specification | Operational Guidance |
|:---|:---|:---|
| **CPU** | 2 to 4 cores (AMD GX-420GI, Ryzen Embedded R1505G/R1606G, Intel) | Thread pool capped to 2 (`HDG_THREADS=2`) to prevent thermal throttling |
| **RAM** | 4 GB to 16 GB DDR4 | Memory-efficient graph (<1.5 GB); 4GB–8GB swapfile required (`vm.swappiness=10`) |
| **GPU** | None (CPU-only execution) | CPU execution with lightweight hidden dimensions (32-dim) |
| **Config** | `configs/hp_thin_client.yaml` | 3 seeds, 50 epochs, 30k domains, batch size 256 |

---

## 2. Session Management & Environment Setup

Because CPU-only evaluations can take hours on embedded cores, **always execute inside a persistent `tmux` session** so network or terminal disconnects never interrupt the pipeline.

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

# 5. Set PyTorch thread allocation
# For HP Thin Clients (ISRO): cap to 2 threads to prevent fanless thermal throttling
export HDG_THREADS=2
# For Enterprise Multi-Core Servers: adjust based on available cores (e.g. 16 or 32)
# export HDG_THREADS=16
```


---

## 3. Step-by-Step Pipeline Execution Sequence

### Step 0: Preflight System & Resource Check
**Script**: `pipeline/00c_preflight.py`  
**Expected Duration**: ~15 seconds  
**Memory**: $< 1$ GB  

Verifies hardware, python dependencies, raw dataset presence, and executes a 1-epoch 10% stratified subsample throughput benchmark with runtime extrapolation.

```bash
python -m pipeline.00c_preflight
```
> **VERIFICATION**: Must print `[PASS] System hardware & dataset checks passed!` Confirm all 4 raw files are detected and estimated runtime is acceptable.

---

### Step 1: Dataset Profiling & Schema Analysis
**Script**: `pipeline/00_profile_dataset.py`  
**Expected Duration**: ~1–2 minutes  
**Memory**: $< 2$ GB (streaming JSON parser via `ijson`)  

Analyzes record counts, key candidate distribution for ASNs and Registrars, and verifies whether temporal rolling-origin evaluation is feasible.

```bash
python -m pipeline.00_profile_dataset
```
> **ACTION**: Review `results/real/profile/profile_report.md` (compact, $\le 200$ lines). Verify that `time_split_valid.json` contains `"time_split_valid": true`.

---

### Step 2: Normalization, Parsing & Time-Matched Benign Cohort
**Script**: `pipeline/01_parse.py`  
**Expected Duration**: ~3–5 minutes  
**Memory**: ~4–8 GB  

Normalizes domain names to e2LD, resolves candidate keys for ASN and Registrar, extracts dual certificate keys (CN and co-issue hash), extracts BGP prefixes, and samples a time-matched benign cohort.

```bash
python -m pipeline.01_parse
```
> **GENERATED ARTIFACTS**:
> - `data/processed/real/domains.parquet` (Complete dataset)
> - `data/processed/real/domains_timematched.parquet` (Time-matched cohort)
> - `data/processed/real/parsing_summary.json`

---

### Step 3: Shortcut & Single-Feature Leakage Audit
**Script**: `pipeline/00b_audit_shortcuts.py`  
**Expected Duration**: ~30 seconds  
**Memory**: ~2 GB  

Audits all domain features for label shortcut proxies. If any single domain feature achieves ROC-AUC $\ge 0.90$, it aborts with a diagnostic error to prevent spurious graph results.

```bash
python -m pipeline.00b_audit_shortcuts
```
> **VERIFICATION**: Must report `Verdict: PASS`. If `subdomain_flag` achieves AUC $\ge 0.90$, confirm `include_has_subdomain: false` in `configs/default.yaml`.

---

### Step 4: Construct Unscaled Base Heterogeneous Graph
**Script**: `pipeline/02_build_graph.py`  
**Expected Duration**: ~1–2 minutes (vectorized extraction)  
**Memory**: ~8–12 GB  

Constructs the unscaled base heterogeneous graph containing domain, ip, nameserver, registrar, asn, and certificate nodes. Standardizers are purposefully NOT fitted globally to prevent test-set data leakage.

```bash
python -m pipeline.02_build_graph
```
> **GENERATED ARTIFACTS**:
> - `data/processed/real/heterodata.pt` (Unscaled base graph)
> - `data/processed/real/id_maps.json` (Topology entity index mappings)

---

### Step 5: Fast Smoke Validation on Real Data
**Script**: `pipeline/run_all.py --smoke`  
**Expected Duration**: ~3–5 minutes  
**Memory**: ~8 GB  

Before committing to a multi-hour run, execute all remaining experimental stages in accelerated smoke mode on the real dataset to verify memory, convergence, and metric calculation end-to-end.

```bash
python -m pipeline.run_all --smoke
```
> **VERIFICATION**: Ensure all stages complete with exit code 0. Confirm `results/real/smoke/REPORT.md` is generated.

---

### Step 6: Full Production Run

You have two equivalent options to execute the complete production evaluation:

#### Option A: Automated One-Command Script (Recommended)
Executes all stages sequentially with atomic run logging, timestamping, error trapping, and automatic results bundling:

```bash
bash pipeline/run_real.sh
```

#### Option B: Step-by-Step CLI Execution
If preferred, you can invoke each experimental module individually. All scripts automatically check `results/real/runs/<name>.json` and skip completed stages if restarted (use `--force` to override):

```bash
# 1. Multi-Seed Baselines (TF-IDF, XGBoost tabular/lexical, MLP no-edges, Length, No-IP)
python -m pipeline.03_run_baselines

# 2. HeteroGNN Multi-Split Training (SAGE, Attn across Random, Time, Group splits)
python -m pipeline.04_train_gnn --variant both

# 3. Systematic Graph & Feature Ablations (35 variants)
python -m pipeline.05_ablation

# 4. Structural Adversarial Attack & Relational GNNGuard Recovery
python -m pipeline.06_attack_eval

# 5. Operational CPU Inference Latency Benchmark
python -m pipeline.07_latency_eval --num-queries 500 --warmup 20

# 6. Dataset Provenance & Shortcut Generalization Checks
python -m pipeline.08_provenance_checks

# 7. Continual Learning & Streaming Domain Updates
python -m pipeline.09_streaming_eval

# 8. BIND Response Policy Zone (RPZ) Rule Emission
python -m pipeline.10_emit_rpz --target-fpr 0.001

# 9. Final Comprehensive Report Compilation
python -m pipeline.11_make_report
```

---

## 4. Packaging & Sending Results Back

Once the run completes, package the final evaluation report, tables, and run logs into an archive:

```bash
# Create compressed results bundle
tar -czvf results_real_bundle_$(date +%Y%m%d).tar.gz \
    results/real/REPORT.md \
    results/real/tables/ \
    results/real/runs/ \
    results/real/rpz/ \
    results/real/logs/
```

### Artifacts to Inspect and Return:
1. `results/real/REPORT.md`: The definitive master document summarizing objectives O1–O5, baseline comparisons, temporal generalization, adversarial recovery, latency, and RPZ blocking rates.
2. `results/real/tables/*.md` and `*.json`: All raw multi-seed tabular evaluations.
3. `results/real/rpz/malicious_domains.rpz`: Production-ready DNS Response Policy Zone rule file.
4. `results/real/logs/*.log`: Execution console logs with timing and system resource statistics.
