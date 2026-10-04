# Operational Guide: Running Malicious-HDG on HP Thin Clients (ISRO Environment)

This operational guide provides deployment instructions, hardware tuning, and Reddit homelab/sysadmin best practices for executing **Malicious-HDG** on **HP Thin Clients** (CPU-only, low-power embedded processors, 4GB–16GB RAM) in ISRO laboratory and workstation environments.

---

## 1. Hardware Context & Constraints

HP Thin Clients (e.g., HP t530, t630, t640, t730, t740) are ultra-compact, power-efficient enterprise desktop appliances commonly deployed across secure institutional labs such as ISRO.

| Specification | Typical HP Thin Client (ISRO Lab) | Operational Implication for PyTorch / GNN |
|:---|:---|:---|
| **CPU Architecture** | 2 to 4 cores (e.g., AMD GX-420GI @ 2.0GHz, Ryzen Embedded R1505G / R1606G, Intel Celeron) | No AVX-512; CPU-bound matrix multiplications require constrained threading. |
| **Cooling & TDP** | Fanless (passive heatsink) or low-RPM 5V blower (15W–35W TDP) | Sustained 100% multi-core load triggers severe thermal throttling (clocks drop ~50%). |
| **RAM** | 4 GB, 8 GB, or up to 16 GB DDR4 | Storing raw datasets + PyG tensors can trigger the Linux Out-Of-Memory (OOM) killer. |
| **Storage** | 16 GB to 128 GB eMMC / M.2 SATA SSD | Free disk space is tight; intermediate parquet and model checkpoints must be lean. |
| **GPU** | None (Integrated display only) | Strict PyTorch CPU-only backend; CUDA/ROCm must be bypassed. |
| **OS** | Linux (Ubuntu 22.04 LTS / Debian 12 / HP ThinPro) | Standard POSIX shell; standard `systemd` and `sysctl` available. |

---

## 2. Reddit Homelab & Sysadmin Community Insights

Discussions on subreddits including `r/homelab`, `r/sysadmin`, and `r/MLQuestions` reveal key challenges and solutions when running Python/PyTorch workloads on HP Thin Clients:

### A. Thread Contention & Thermal Throttling
* **The Problem**: Default PyTorch / OpenMP / BLAS behavior detects all logical threads (e.g. 4 threads) and runs at 100% saturation. On fanless or compact thin client chassis, thermal dissipation cannot keep up, resulting in severe thermal downclocking, latency spikes, or system lockups.
* **The Fix**: Cap thread execution to **2 threads** (`export HDG_THREADS=2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`). This leaves headroom for OS background processes and keeps package temperatures within passive cooling limits.

### B. Linux Kernel Out-Of-Memory (OOM) Killer
* **The Problem**: On 4GB–8GB RAM thin clients, loading the complete uncompressed DomainRadar dataset (~930k JSON records) or running GNN forward/backward passes with batch size 1024 causes physical RAM exhaustion. The kernel triggers `Out of memory: Killed process python` (`exit code 137`).
* **The Fix**:
  1. Configure a dedicated **4 GB to 8 GB swapfile** with low swappiness (`vm.swappiness=10`).
  2. Use streaming JSON parsing via `ijson` and cap domain population to **30,000 domains** (`max_domains: 30000`, `max_per_class: 15000`).
  3. Reduce model hidden dimensionality from 64 to 32 (`hidden_dim: 32`, `out_dim: 16`), cutting activation memory by >50%.
  4. Reduce GNN batch size to **256** and enforce explicit garbage collection (`gc.collect()`) after training iterations.

---

## 3. Fast Hardware Probing & ChatGPT Calibration Prompt

Before running or editing anything, run the automated diagnostic script directly on your ISRO workstation terminal:

```bash
bash tools/check_system.sh
# (Or using Python: python3 tools/probe_hardware.py)
```

### Ready-to-Use ChatGPT Prompt Template
You can copy the terminal output and paste it into ChatGPT with this prompt:

> **ChatGPT Prompt**:
> *"I am running the Malicious-HDG GNN pipeline on an HP Thin Client at ISRO. Here is the hardware diagnostic output from running `bash tools/check_system.sh`:*
> *```*
> *[PASTE OUTPUT OF tools/check_system.sh HERE]*
> *```*
> *Based on my CPU cores, physical RAM, and swap space, tell me the exact line edits to make to `configs/hp_thin_client.yaml` so the training completes smoothly without running out of memory (OOM) or overheating."*

---

## 4. Host System Preparation (Run Once on HP Thin Client)

Before starting the pipeline, execute these setup steps on the HP Thin Client host terminal:

### Step 1: Create and Activate a Swapfile (Mandatory for < 16GB RAM)
Check existing swap with `free -h` or `swapon --show`. If swap is 0 or less than 4GB, create a swapfile:


```bash
# 1. Allocate 4 GB swapfile (use 8G if disk space permits)
sudo fallocate -l 4G /swapfile

# 2. Restrict permissions to root only
sudo chmod 600 /swapfile

# 3. Format as Linux swap
sudo mkswap /swapfile

# 4. Enable swap immediately
sudo swapon /swapfile

# 5. Persist swap in /etc/fstab across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 6. Optimize swappiness (prevents aggressive swapping until RAM is truly exhausted)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

Verify with:
```bash
free -h
```

---

## 4. Configuration: `configs/hp_thin_client.yaml`

The repository provides a dedicated configuration profile specifically tuned for HP Thin Clients:

```yaml
hardware:
  profile: "hp_thin_client"
  num_threads: 2               # Capped at 2 threads for thermal stability

parsing:
  max_per_class: 15000         # Prevents stream parser memory exhaustion

graph:
  max_domains: 30000           # Peak in-memory graph < 1.5 GB
  hub_degree: 300              # Prunes super-dense hub nodes

model:
  sage:
    hidden_dim: 32             # Cuts tensor memory by 50%
    out_dim: 16
  attn:
    hidden_dim: 32
    out_dim: 16
    num_heads: 2
  training:
    epochs: 50                 # Fast convergence, protects low-clock CPU
    patience: 10
    batch_size: 256            # Safe memory budget per gradient step

baselines:
  tfidf:
    max_features: 2500         # Lean sparse matrix representation
  xgboost:
    n_estimators: 50           # Shallow, fast decision trees
    max_depth: 4

evaluation:
  bootstrap_samples: 200       # Fast bootstrap CI computation on dual-core CPU
  evaluation_seeds: [42, 1337, 2024] # 3 robust evaluation seeds
```

---

## 5. Step-by-Step Execution Sequence

### Step 0: Start a Persistent Session
HP Thin Client embedded CPUs take longer than multi-core servers. **Always use `tmux`** so network or terminal disconnects never terminate your pipeline:

```bash
# Start or attach to tmux
tmux new -s isro_hdg

# Activate python virtual environment
cd /path/to/Malicious-HDG
source venv/bin/activate
```

---

### Step 1: Preflight Verification
Run preflight with the `--thin-client` flag:

```bash
python -m pipeline.00c_preflight --thin-client
```

> **Expected Verification**:
> - Prints `[HP THIN CLIENT / RESOURCE-CONSTRAINED PROFILE DETECTED]`.
> - Confirms PyTorch threads = 2 and swap memory is available.
> - Reports `[PASS] System hardware & dataset checks passed!`.

---

### Step 2: Automated Pipeline Execution

You have two primary options:

#### Option A: One-Command Shell Script (Recommended)
Automatically selects the HP Thin Client profile, sets thread pools, validates swap, skips already completed runs, and compiles results:

```bash
bash pipeline/run_real.sh --thin-client
```

#### Option B: Fast End-to-End Smoke Validation
Run the full 13-stage pipeline in accelerated smoke mode to verify everything end-to-end in ~5 minutes:

```bash
python -m pipeline.run_all --thin-client --smoke
```

#### Option C: Step-by-Step Modular Execution
Execute individual stages at your own pace (each automatically skips if already completed):

```bash
# 1. Dataset Profiling
python -m pipeline.00_profile_dataset --config configs/hp_thin_client.yaml

# 2. Parsing & Time-Matched Benign Cohort
python -m pipeline.01_parse --config configs/hp_thin_client.yaml

# 3. Shortcut Audit
python -m pipeline.00b_audit_shortcuts --config configs/hp_thin_client.yaml

# 4. Build Unscaled Base HeteroData Graph
python -m pipeline.02_build_graph --config configs/hp_thin_client.yaml

# 5. Multi-Seed Baselines (TF-IDF, XGBoost, MLP)
python -m pipeline.03_run_baselines --config configs/hp_thin_client.yaml

# 6. HeteroGNN Evaluation (SAGE & Attn across Random, Time, Group splits)
python -m pipeline.04_train_gnn --config configs/hp_thin_client.yaml --variant both

# 7. Topological Ablation Study
python -m pipeline.05_ablation --config configs/hp_thin_client.yaml

# 8. Structural Adversarial Attack & GNNGuard Recovery
python -m pipeline.06_attack_eval --config configs/hp_thin_client.yaml

# 9. Operational CPU Latency Benchmark
python -m pipeline.07_latency_eval --config configs/hp_thin_client.yaml --num-queries 200 --warmup 10

# 10. Provenance Checks
python -m pipeline.08_provenance_checks --config configs/hp_thin_client.yaml

# 11. Streaming Continual Learning
python -m pipeline.09_streaming_eval --config configs/hp_thin_client.yaml

# 12. BIND RPZ Rule Emission
python -m pipeline.10_emit_rpz --config configs/hp_thin_client.yaml --target-fpr 0.001

# 13. Compile Master Report
python -m pipeline.11_make_report --config configs/hp_thin_client.yaml
```

---

## 6. Live Monitoring on HP Thin Client

In a second terminal or tmux window, monitor system health during execution:

```bash
# Monitor memory and swap in real-time
watch -n 2 'free -m'

# Monitor CPU temperatures and frequencies
htop
```

If CPU temperatures rise above 80°C or clock frequencies drop below 1.2GHz, verify that the thin client is positioned vertically with unobstructed ventilation grilles.

---

## 7. Packaging & Delivering Results

When completed, package the final evaluation deliverables:

```bash
tar -czvf results_isro_thin_client_$(date +%Y%m%d).tar.gz \
    results/real/REPORT.md \
    results/real/tables/ \
    results/real/runs/ \
    results/real/rpz/ \
    results/real/logs/
```
