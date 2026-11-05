# GPU Adaptation & Cluster Execution Plan: NVIDIA A100 80GB (Phase 2 & Beyond)

**Date**: November 5, 2026  
**Status**: APPROVED PROTOCOL SPECIFICATION (Incorporating Reviewer Feedback)  
**Target Hardware**: NVIDIA A100-SXM4-80GB (Remote Shared Compute Node) | CPU Reference (ISRO 48-Core Server / Apple Silicon Workstation)  
**Applicability**: Zenodo DomainRadar v2 (100k Benchmark Cohort & Full-Corpus Extension)  
**Companion Documents**: [`docs/analysis_plan.md`](file:///Users/urvagandhi/Developer/Personal/Krisha/Malicious-HDG-self/docs/analysis_plan.md), [`docs/ENGINEERING_DECISIONS.md`](file:///Users/urvagandhi/Developer/Personal/Krisha/Malicious-HDG-self/docs/ENGINEERING_DECISIONS.md)

---

## 1. Executive Summary & Reviewer Conditions

Following formal reviewer review, the GPU adaptation for Malicious-HDG is approved subject to **nine mandatory constraints**. This document formalizes these rules, the software verification protocol, memory safeguards for shared environments, and the phase gate structure.

### The Nine Mandatory GPU Constraints

| # | Domain | Constraint & Specification | Implementation & Safeguard |
|:---:|:---|:---|:---|
| **1** | **Software Stack & Protocol Duality** | **Explicit Environment Duality**: The CPU reference benchmark was trained on the ISRO 48-core server under **PyTorch 2.14.0+cu130 and PyG 2.8.0.post1**. The remote A100 GPU node hosts **PyTorch 2.5.1 + CUDA 12.1 and PyG 2.7.0**. Because environments differ, GPU runs are designated as an independent execution protocol (`results/real_gpu/`). Phase 2a must verify Seed 42 parity within $\pm 0.002$ AUC before launching full multi-seed runs. | Probe host environment via `tools/check_system.sh` (`torch.__version__`, `torch.version.cuda`, `nvidia-smi`). |
| **2** | **Default Device** | **Default Strictly to `cpu`**: Never default to `auto` or silently switch device. | CLI defaults to `--device cpu`. GPU execution requires explicit `--device cuda` or `HDG_DEVICE=cuda`. Run JSONs must record `device`, `torch_version`, `cuda_version`, `git_commit`, and `hostname`. |
| **3** | **Precision (No BF16/AMP)** | **Strict FP32 / Disable TF32**: `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`. | The 100k heterogeneous graph fits in ~3.5 GB VRAM. BF16's 8-bit mantissa creates logit ties near critical classification thresholds ($\sim 2.6$), collapses tail metrics at $\text{FPR} \le 0.1\%$, breaks GNNGuard cosine pruning ($\varepsilon = 10^{-8}$), and invalidates partial AUC ranking margins. |
| **4** | **Run Isolation (No Run Mixing)** | **Dedicated Result Directory**: Separate GPU runs from CPU runs. Never allow `should_skip_run` to reuse CPU artifacts on GPU. | All GPU runs output to `results/real_gpu/` or affix `_gpu` to run file identifiers. CPU and GPU evaluations must never be mixed within paired statistical tests. |
| **5** | **Determinism & Scatter** | **Acknowledge CUDA Non-Determinism**: PyG CUDA scatter-add operations are atomic and inherently non-deterministic across executions. | Set `torch.use_deterministic_algorithms(True, warn_only=True)`. Document in the thesis that exact bit-for-bit seed reproducibility applies within CPU reference runs, while GPU runs exhibit bounded floating-point scatter variance. |
| **6** | **Attack Tensor Placement** | **Generate on CPU, Transfer Atomically**: Structural perturbations build adjacency lists on host memory. | `generate_structural_attack` constructs edge index tensors on CPU, then performs atomic `.to(device)` transfer to prevent split-device graph corruptions. |
| **7** | **Latency Isolation** | **O4 is Strictly a CPU Claim**: Production recursive DNS filtering operates on commodity server CPUs without dedicated GPUs. | `pipeline/07_latency_eval.py` remains 100% CPU-only. Latency Pareto benchmarks are never measured on GPU to avoid misleading production claims. |
| **8** | **Shared GPU Safeguards** | **Multi-Tenant Protection**: Prevent out-of-memory collisions on shared A100 nodes. | Enforce `CUDA_VISIBLE_DEVICES`, configure `torch.cuda.set_per_process_memory_fraction(0.35)` (~28 GB cap), and re-check `torch.cuda.mem_get_info()` immediately prior to each seed run. |
| **9** | **Pipeline Harmonization** | **Complete Stage Coverage in `run_real.sh`**: Runner scripts must execute all evidence closure stages. | Runner scripts explicitly sequence: certified low-FPR calibration, purity-ranked structural attacks with 3 differences, bounded-sampling latency Pareto frontier, cross-source transfer audits, and pre-committed stop rules for stacker/pAUC. |

---

## 2. Memory & Compute Budget on NVIDIA A100 (80GB)

### 2.1 Graph Footprint Analysis (100k Cohort)
- **Node Count**: 348,294 heterogeneous nodes across 6 types (`domain`: 100,000, `ip`: ~85,000, `nameserver`: ~42,000, `asn`: ~3,500, `registrar`: ~1,200, `cert`: ~116,000).
- **Edge Count**: ~1.2 million bidirectional edges across 12 relational types.
- **Node Feature Tensors**:
  - Domain features: 5 float32 scalars ($100,000 \times 5 \times 4\text{ bytes} \approx 2.0\text{ MB}$).
  - Adjacency / Edge Indices: $1.2\text{M} \times 2 \times 8\text{ bytes} \approx 19.2\text{ MB}$.
  - Model Hidden Dimensions: $h = 64, L = 2$ layers ($\sim 150\text{ MB}$ activations).
- **Total Peak VRAM Under Full-Batch FP32 Training**: **$\approx 3.2 - 3.8\text{ GB}$**.
- **Available Headroom**: On an 80 GB card, $> 75\text{ GB}$ remains free. Neighbor sampling or graph mini-batching is **unnecessary during training**; full-batch transductive training fits with 20x memory margin.

### 2.2 Shared-Server Concurrency Protection
Because the ISRO server is shared across researchers:
```python
import torch

def setup_shared_gpu_environment(dev_idx: int = 0, memory_fraction: float = 0.35, min_free_vram_gb: float = 8.0):
    """
    Enforces deterministic FP32 operation, memory limits, and pre-flight VRAM verification.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Explicit --device cuda was requested.")
    
    # 1. Enforce strict FP32 precision (Disable TF32 for tail metric fidelity)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 2. Set deterministic scatter mode
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 3. Dynamic pre-seed VRAM verification
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_free_vram_gb:
        raise RuntimeError(
            f"Insufficient VRAM on CUDA device {dev_idx}: {free_gb:.2f} GB free, "
            f"minimum required: {min_free_vram_gb:.2f} GB. Aborting to avoid OOM collision."
        )
        
    # 4. Cap per-process memory allocation to protect other researchers
    torch.cuda.set_per_process_memory_fraction(memory_fraction, dev_idx)
```

---

## 3. Implementation Blueprint

### 3.1 Device Resolver Module (`src/hdg/device.py`)
Provides centralized, auditable device resolution across all CLI tools:
- Resolves `'cpu'`, `'cuda'`, or `'cuda:N'`.
- Defaults strictly to `'cpu'` when unsupplied.
- Emits environment provenance metadata (`torch_version`, `cuda_version`, `device_name`, `git_commit`) into every output JSON.

### 3.2 Standalone Pre-Flight Script (`tools/verify_gpu_graph.py`)
Before launching production training on the cluster, Krisha will execute:
```bash
python tools/verify_gpu_graph.py --device cuda
```
Verification checks executed:
1. Driver & CUDA Toolkit: confirms `torch.cuda.is_available()` and driver compatibility.
2. Tensor Allocation: allocates 4 GB dummy tensor on `cuda:0` and confirms memory fraction enforcement.
3. FP32 Precision Check: verifies `allow_tf32 == False`.
4. Synthetic HeteroGNN Pass: tests forward + backward pass on synthetic `HeteroData` with SAGE, ATTN, and GNNGuard (cosine similarity $\varepsilon = 10^{-8}$).
5. Real Graph Profile: loads `data/processed/real/heterodata.pt`, transfers to GPU, records exact peak VRAM, and tests 1 training epoch.

### 3.3 Training & Ablation Execution (`pipeline/04_train_gnn.py`, `05_ablation.py`)
- CLI flag `--device` added (default: `cpu`).
- When `--device cuda` is specified:
  - Output directory defaults to `results/real_gpu/` to guarantee zero contamination of frozen CPU baselines in `results/real/`.
  - Checkpoints are saved under `artifacts/checkpoints/real_gpu/`.
  - Manifest dictionary records hardware metadata in `runs/*.json`.

---

## 4. Phase-by-Phase Roadmap

### Phase 0: Freeze & Provenance (Completed)
- [x] Tag CPU benchmark results as `cpu-benchmark-v1`.
- [x] Freeze `docs/analysis_plan.md` in git before executing post-hoc evaluations.
- [x] Make `results/real/` strictly read-only reference data.

### Phase 1: Evidence Closure on CPU (Completed & Verified)
- [x] **Corrected O5 Attack**: Purity-ranked targets ($\ge 99\%$ benign purity); reported via 3 differences ($\Delta_{\text{att}}, \Delta_{\text{res}}, \Delta_{\text{def}}$) with 95% CIs and realized FPR bound ($0.385\% \to 0.112\%$).
- [x] **Latency Pareto**: Bounded neighbor sampling $[k_1, k_2]$ Pareto table; ego build vs forward pass breakdown; logit cosine agreement ($> 0.99$), AUC, and certified TPR.
- [x] **Reconciled Numbers**: Tabular XGBoost precision discrepancy resolved (Seed 42 = 31.88% vs 5-seed mean = 21.83%) with mathematical definitions.
- [x] **ATTN Seed 777 Diagnosis**: Reframed as "model selection failed under group shift (validation-test gap 0.21)" with per-seed trajectory audit.
- [x] **Attributions & Citations**: AO-GNN cited as Huang et al. (WWW 2022, AUC-oriented loss); 50 CESNET discrepancy noted as unexplained; VirusTotal verification cited (*Data in Brief* Sec 2.1); `batch_size: 1024` clarified as legacy/unused.
- [x] **Strongest Baseline Paired Comparison**: TF-IDF n-gram Logistic Regression ($0.8992 \pm 0.0016$) established as strongest baseline; paired t-test against HeteroGNN (+0.0222, $p=0.0026$).
- [x] **Cross-Source Transfer Benchmark**: XGBoost (0.4690) and TF-IDF (0.4984) collapse to chance; HeteroGNN retains 0.6209 (+0.1225 over TF-IDF).
- [x] **100% Dynamic Report**: `pipeline/11_make_report.py` completely refactored to read from JSON artifacts or compute dynamically with zero static mock tables.

### Phase 2: One GPU Protocol (Ready to Launch)
1. **Phase 2a: Environment & Device Parity Verification**:
   - Probe A100 environment versions (`torch.__version__`, `torch.version.cuda`, `nvidia-smi`) and log to `results/real_gpu/env_manifest.json`.
   - Run device-parity benchmark on Seed 42 (CPU vs GPU in same environment); verify test ROC-AUC agreement within $\pm 0.002$.
   - Measure 1-epoch training time and peak VRAM.
2. **Phase 2b: Protocol Tuning & Co-Primary Hyperparameter Selection**:
   - Selection metric: **Co-primary evaluation on in-distribution validation pAUC ($\text{FPR} \le 1\%$) AND cross-source transfer AUC (Variant 1 / LOMSO)** on Seed 42. Selecting exclusively on in-distribution validation is strictly prohibited to prevent selecting models that exploit collection-source artifacts.
   - Early stopping & epoch floor: Enforce a **minimum 50-epoch floor** and expand patience to **60 epochs** to eliminate the premature stopping failure diagnosed on Seed 777.
   - Multi-Seed Execution: Re-run HeteroGNN (SAGE & ATTN) across all 5 seeds on GPU (`--device cuda`), and re-run baselines under identical protocol.
   - Save strictly to `results/real_gpu/`; keep `results/real/` (CPU) as reference.

### Phase 3: Scale-Up Study (Full Zenodo Corpus)
- Run parsed corpus without cohort capping (`max_domains: null`, ~250k domains).
- Expands test negatives from ~10k to ~25k, quadrupling statistical resolution at $\text{FPR} = 0.10\%$.
- Reported as an independent scalability study in the dissertation; the 100k benchmark remains the primary frozen headline.

### Phase 4: Architectural Refinements
1. **Lexical Node Integration**: Ingest char n-gram TF-IDF / SVD embeddings directly into domain node attributes, closing the lexical validity gap.
2. **Source-Robust Invariant Training**: Group-balanced loss across benign collections (Umbrella vs CESNET) to mitigate source leakage shortcuts.
3. **Relation-Attention GNN**: Deploy relation-specific attention if group-split evaluation shows persistent partition-dependent gaps.

---

## 5. Summary of Pre-Flight Commands for Cluster / Krisha

When accessing the A100 server via SSH:
```bash
# 1. Inspect GPU hardware and driver status
nvidia-smi

# 2. Verify Python environment and PyTorch CUDA linkage
python3 -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 3. Run standalone GPU verification script (Stage 2 check)
python3 tools/verify_gpu_graph.py --device cuda

# 4. Launch Phase 2 GPU training (FP32, isolated results directory)
python3 -m pipeline.04_train_gnn --config configs/active_config.yaml --device cuda --out-dir results/real_gpu
```
