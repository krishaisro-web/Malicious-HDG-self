# ISRO Reproducibility & Deployment Guide

This guide provides step-by-step instructions for running, validating, and resuming the Malicious-HDG pipeline on an ISRO Linux CPU environment (e.g., HP Thin Client or enterprise compute server).

---

## 1. Prerequisites & Environment Setup

### System Requirements
- **OS**: Linux (CentOS/RHEL/Ubuntu/Debian) or macOS.
- **Python**: Python 3.10, 3.11, or 3.12 (Python 3.12 recommended).
- **Compute**: CPU-only (CUDA not required; all models and PyTorch Geometric layers execute natively on CPU).

### Safe Git Clone & Branch Checkout
On the ISRO machine, clone the repository and check out the intended implementation branch:

```bash
git clone https://github.com/krishaisro-web/Malicious-HDG-self.git
cd Malicious-HDG-self
git checkout main
git status
```

Verify that the working tree is clean and matches the approved Git commit.

### Virtual Environment Creation
Create an isolated Python virtual environment and install the locked dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

### CPU-Only PyTorch & PyTorch Geometric Verification
Verify that PyTorch and PyTorch Geometric import cleanly and utilize CPU multi-threading safely:

```bash
python -c "
import torch
import torch_geometric
print('PyTorch Version:', torch.__version__)
print('PyG Version:', torch_geometric.__version__)
print('CUDA Available:', torch.cuda.is_available(), '(Expected False for CPU deployment)')
"
```

---

## 2. Dataset Verification & Checksum Audit

### Placing the Raw Zenodo Dataset
Place the raw Zenodo DomainRadar v2 JSON files in `data/raw/zenodo/`:
- `data/raw/zenodo/malware.json`
- `data/raw/zenodo/benign_umbrella.json`
- `data/raw/zenodo/benign_cesnet.json`

### Validating Against the Data Manifest
Run the preflight audit to verify dataset integrity, required columns, and CPU/RAM resources:

```bash
python -m pipeline.00c_preflight
```

If dataset files are missing, the preflight script immediately fails with actionable instructions.

---

## 3. Selecting the Appropriate Configuration Profile

The repository supports two primary hardware profiles:

1. **Standard / High-Memory Server** (`configs/default.yaml`):
   - Used when $\ge 16$ GB RAM is available.
   - Evaluates full datasets with standard thread allocations.
2. **HP Thin Client / Constrained CPU** (`configs/hp_thin_client.yaml`):
   - Optimized for 2–4 CPU cores and 4–16 GB RAM.
   - Caps PyTorch threads to 2 (`HDG_THREADS=2`), uses smaller batch sizes (256/128), and prevents fanless chassis thermal throttling.

To activate the thin client profile, simply pass `--thin-client` or `--config configs/hp_thin_client.yaml`.

---

## 4. Execution Workflow

### Step 1: Pre-Execution Software Test (Fixture Validation)
Before executing expensive operations on raw data, run the comprehensive unit test suite:

```bash
pytest tests/ -v
```

All 54 tests must pass.

### Step 2: Running the Complete Pipeline
Execute the full automated end-to-end pipeline:

```bash
# Option A: Standard Enterprise / Server execution
python -m pipeline.run_all --force

# Option B: HP Thin Client / Constrained CPU execution
python -m pipeline.run_all --thin-client --force

# Option C: Quick diagnostic smoke test (safe, isolated in results/smoke/)
python -m pipeline.run_all --smoke
```

The pipeline automatically coordinates all 11 stages:
1. `00c_preflight.py`: Environment and raw dataset verification.
2. `00_profile_dataset.py`: Longitudinal dataset profiling and timestamp diagnostics.
3. `01_parse.py`: Schema normalization, streaming parsing, and collision resolution.
4. `00b_audit_shortcuts.py`: Single-feature predictive shortcut audit.
5. `02_build_graph.py`: Cumulative base graph construction and immutable cohort bundling.
6. `03_run_baselines.py`: TF-IDF, XGBoost tabular, and single-feature benchmark baselines.
7. `04_train_gnn.py`: SAGE and ATTN HeteroGNN evaluation across `random`, `time`, and `group` splits.
8. `05_ablation.py`: Node, edge, and lexical feature-group ablations.
9. `06_attack_eval.py`: GNNGuard adversarial robustness evaluations.
10. `07_latency_eval.py`: Inference latency and throughput benchmarks.
11. `08_provenance_checks.py`: Source disparity and collection shortcut audits.
12. `09_streaming_eval.py`: Continual learning and streaming update evaluation.
13. `10_emit_rpz.py`: Fail-safe DNS Response Policy Zone rule generation.
14. `11_make_report.py`: Comprehensive synthesis report generation (`results/real/REPORT.md`).

---

## 5. Resuming an Interrupted Training Run

If a training job is interrupted (e.g., system reboot, power outage, terminal disconnect), **simply rerun the command without `--force`**:

```bash
python -m pipeline.run_all --thin-client
```

### Resumption Behavior:
- **Completed Stages**: Skipped instantly because existing output artifacts match the cryptographic experiment fingerprint.
- **Interrupted GNN Training**: `src/hdg/training/train.py` detects the atomic epoch-level checkpoint in `artifacts/checkpoints/real/gnn_<variant>_<split>_seed_<seed>.pt`, verifies that the experiment fingerprint matches, restores the model weights, optimizer, learning-rate scheduler, and best validation score, and resumes seamlessly from epoch $N+1$.

---

## 6. Artifact Transfer Between ISRO and Development Machines

Production datasets (`data/raw/`, `data/processed/real/`), checkpoints (`artifacts/checkpoints/real/*.pt`), and raw logs are gitignored and must **not** be pushed to GitHub.

To package and transfer approved evaluation tables, manifests, and synthesis reports:

```bash
# Bundle approved results without large weights
tar -czvf isro_evaluation_results.tar.gz \
  results/real/REPORT.md \
  results/real/tables/ \
  results/real/profile/ \
  results/real/rpz/
```

Transfer `isro_evaluation_results.tar.gz` via approved external media or secure file transfer protocols.

---

## 7. Synchronizing Code Changes Back to GitHub

To contribute source code or documentation updates back from the ISRO machine:

```bash
# Verify only tracked source code and docs are modified
git status --short

# Commit approved code modifications
git add src/ pipeline/ tests/ configs/ docs/ pyproject.toml .gitignore
git commit -m "feat(pipeline): reproducibility and incremental execution updates for ISRO CPU environment"

# Push to the approved GitHub repository main branch (never force-push)
git push origin main
```
