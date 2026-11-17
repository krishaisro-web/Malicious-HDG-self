# Malicious-HDG Agent Operational Instructions & Persistent Memory

> **CRITICAL MANDATE**: Any AI agent or developer continuing this work **MUST READ THIS ENTIRE FILE FIRST** before inspecting other files, diagnosing issues, or running pipeline commands. All operational context, hardware boundaries, memory math, and execution protocols are codified here.

---

## 1. Target Execution Cluster & Environment Profile

- **Cluster Node**: ISRO Thin Client Server `odpd-lcws-a100-2024-01` (RHEL 9.4 Plow, Kernel 5.14.0-427.13.1.el9_4.x86_64).
- **CPU**: 128 logical threads (64 physical cores), Dual-Socket Intel(R) Xeon(R) Gold 6430 @ 3.4 GHz.
- **Physical RAM**: 503.57 GB total (468 GB available). Swap: 64.0 GB (OOM killer protection active).
- **User & Active Path**: `rohit` @ `/home/rohit/Malicious-HDG-self`.
- **Python / Conda Environment**: `zeekGNN` (`/home/rohit/miniconda3/envs/zeekGNN/bin/python`, Python 3.10.20, PyTorch 2.5.1+cu121, CUDA 12.1).
- **Storage Partitions**:
  - `/home` partition: `/home/rohit/Malicious-HDG-self` (~24 GB free, used for code, checkpoints, and lightweight results).
  - `/data` secondary partition: `/data/rohit-Malicious-HDG` (**17.6 TB free**, dedicated for large artifacts/raw datasets).
- **GPU Hardware**:
  - **GPU 0**: NVIDIA A100-80GB PCIe (Total VRAM: 79.25 GiB / 81,155 MiB).
  - **GPU 1**: NVIDIA T400-4GB PCIe (Protected; explicitly bypassed via `CUDA_VISIBLE_DEVICES=0`).

---

## 2. Absolute Boundaries & Co-Tenant Safeguards

1. **Co-Tenant Daemon Safety (PID 57310)**:
   - Command: `vllm serve QuantTrio/Qwen3.5-27B-AWQ --served-model-name gpt-oss-120b --gpu-memory-utilization 0.90 --max-model-len 65536 --port 8001`
   - Uptime: >1,531 hours (>63 days).
   - Memory Occupied: **~70.49 GiB (72,180 MiB)**.
   - **RULE**: **NEVER KILL, PAUSE, OR DISRUPT PID 57310**. It is a shared production LLM inference engine.
   - **CRITICAL NOTE ON vLLM**: vLLM pre-allocates 90% of GPU VRAM as a persistent KV-cache pool at startup. **It will NEVER finish or release memory naturally.** We must work strictly within the remaining free VRAM.
2. **Available VRAM Budget**:
   - Total VRAM: **79.25 GiB**
   - vLLM Allocation: **70.49 GiB**
   - **Strictly Free Available VRAM for our pipeline: ~8.74 to 8.76 GiB (8,947 to 8,970 MiB)**.
3. **Phase 2 Research Protocol Rules**:
   - **Strict FP32**: `allow_tf32 = False` in PyTorch matmul and cuDNN. Zero BF16, zero FP16 mixed precision. Prevents logit ties at the operational threshold and preserves GNNGuard cosine edge pruning ($eps = 10^{-8}$).
   - **Output Isolation**: Output writes exclusively to `results/real_gpu/` and `artifacts/checkpoints/real_gpu/`. `results/real/` is frozen reference data.
   - **Hardware Invariance**: Latency evaluation (Stage 9) strictly runs on CPU (`--device cpu`) to maintain DNS production serving validity.

---

## 3. Dataset Scale & Exact VRAM Footprint (1 Lakh Domains)

- **Dataset Scale**: Exactly **100,000 domains (1 Lakh / 1L)** extracted from the real Zenodo + DNS cohort.
- **Heterogeneous Graph Scale**:
  - `domain` nodes: 100,000 (22 features)
  - `ip` nodes: 158,237 (10 features)
  - `ns` (nameserver) nodes: 72,000 (10 features)
  - `as` (autonomous system) nodes: 18,000 (10 features)
  - **Total Graph Nodes**: **348,237 nodes**
  - **Total Graph Edges**: **1,369,520 edges** across 6 canonical relation types.

### Detailed Tensor Memory Math on the A100

| Component | Dimensions / Counts | Precision | Exact Memory Footprint |
| :--- | :--- | :--- | :--- |
| **Domain Nodes Tensor** | $100,000 \times 22$ | FP32 | **8.80 MB** |
| **IP Nodes Tensor** | $158,237 \times 10$ | FP32 | **6.33 MB** |
| **Nameserver Nodes Tensor** | $72,000 \times 10$ | FP32 | **2.88 MB** |
| **Autonomous System Tensor** | $18,000 \times 10$ | FP32 | **0.72 MB** |
| **Graph Topology (Edge Indices)**| $1,369,520 \times 2$ | int64 | **21.91 MB** |
| **Total Static Graph in GPU VRAM**| **348,237 nodes, 1.37M edges** | — | **~40.64 MB** |
| **Model Weights (HeteroGNN)** | Linear Proj + 2 Conv Layers + Head | FP32 | **~1.50 MB** |
| **Adam Optimizer State** | First & second momentum buffers | FP32 | **~3.00 MB** |
| **SAGE Conv Peak Activation** | 200 epochs across 1.37M edges | FP32 | **~1.80 GB** *(Passed: 0.61s/ep)* |
| **ATTN (HGTConv) Peak Activation**| $Q, K, V$, Attn Weights $\alpha$, Messages | FP32 | **~4.43 GB** *(Direct forward/backward)* |

### Headroom Analysis on ISRO Server
- **Available Free VRAM on GPU**: **8.76 GiB (8,970 MiB)**
- **Malicious-HDG Peak Required VRAM**: **4.43 GiB (4,536 MiB)**
- **Safety Headroom Remaining**: **+4.33 GiB (4,434 MiB) [49.4% Buffer]**

---

## 4. History of Errors, Root Causes & Permanent Patches

### Fix 1: PyG Scatter Fallback Allocation Spike (1.33 GiB)
- **Problem**: When PyG C++ extension (`torch_scatter`) is missing, fallback `scatter()` broadcasts 1D indices to 2D `[1369520, 64]` int64 tensors (701 MB) and allocates a 1.33 GiB workspace in `scatter_add_`.
- **Patch (Commit `c019ac6`)**: Implemented `apply_memory_efficient_scatter_patch()` in `src/hdg/device.py` using PyTorch native `index_add_` and `index_reduce_`. Completely eliminated the 701 MB broadcast tensor and the 1.33 GiB workspace allocation with exact mathematical parity (diff = 0.0).

### Fix 2: HGTConv Activation Checkpointing Spike (336 MiB OOM)
- **Problem**: In commit `805b7b2`, `torch.utils.checkpoint.checkpoint` was applied to `HGTConv`. In a 2-layer GNN, recomputing Layer 1 forward during the backward pass while holding Layer 2 backward gradients accumulated **7.52 GiB** in the CUDA cache. When `index_add_` backward ran `grad_out.index_select(0, index)` for 336.00 MiB, only 328.62 MiB was free, failing by **7.38 MiB**.
- **Patch (Commit `037b41b`)**:
  - Removed `checkpoint()` from `HeteroGNN.forward` in `src/hdg/models/hetero_gnn.py`. Direct layer execution lets PyTorch free intermediate activations as backward progresses, reducing peak memory from 7.52 GB to **4.43 GB**.
  - Enforced `optimizer.zero_grad(set_to_none=True)` in `src/hdg/training/train.py`.
  - Added explicit garbage collection, `model.zero_grad(set_to_none=True)`, and `torch.cuda.empty_cache()` inside and after `lr_grid`.
  - Added CUDA memory flushes between ablations (`pipeline/05_ablation.py`) and attacks (`pipeline/06_attack_eval.py`).

### Fix 3: Multi-Seed Resumption across Git Commits
- **Problem**: Pulling bug fixes changes the Git HEAD hash, which previously invalidated experiment fingerprints and forced re-running already completed stages (e.g. 10 minutes of SAGE).
- **Patch (Commit `037b41b`)**: Updated `pipeline/04_train_gnn.py` to inspect existing run JSONs in `results/real_gpu/runs/`. If a run completed successfully with full metrics (`"status": "SUCCESS"` and `"roc_auc"`), it is automatically skipped in 0.1 seconds, resuming straight to the pending variant (ATTN).

### Fix 4: Cross-Split Broadcasting Mismatch in Paired Comparisons (15,001 vs 53,943)
- **Problem**: In Stage 6b (Temporal Split), after SAGE and ATTN finished training all epochs across all seeds, `pipeline/04_train_gnn.py:459` attempted to compute paired McNemar and bootstrap comparisons against `baseline_xgboost_tabular` (Stage 5). However, XGBoost was evaluated on the random split test set (15,001 domains), while temporal models were evaluated on the chronological holdout (53,943 domains). NumPy raised `ValueError: operands could not be broadcast together with shapes (15001,) (53943,)`.
- **Patch (Commit `f48845d`)**:
  - In `pipeline/04_train_gnn.py`: Added explicit guard `if args.split_type == "random":` for the tabular baseline paired comparison.
  - In `src/hdg/metrics.py`: Added defensive length checks in `paired_mcnemar_test`, `paired_bootstrap_auc_difference`, and `multi_seed_paired_comparisons` to gracefully bypass incompatible evaluation split dimensions without crashing.

### Fix 5: Comprehensive Pyflakes Static Analysis & Variable Hygiene Audit
- **Problem**: Comprehensive pyflakes code analysis across `pipeline/`, `src/`, `tools/`, and `tests/` revealed latent runtime name errors and code hygiene issues that would crash later stages during long-running execution:
  1. `pipeline/01_parse.py:53`: `json.load()` called without `import json`.
  2. `pipeline/07_latency_eval.py:267`: Type hint `List[float]` evaluated at runtime without `from typing import List`.
  3. `pipeline/11_make_report.py:487`: `compute_prevalence_adjusted_precision` called without import from `src.hdg.metrics`.
  4. `pipeline/11_make_report.py:1187`: Unbound variable `res_dir` referenced instead of `paths.results_dir`.
  5. `src/hdg/eval/streaming.py:148`: `optimizer.zero_grad()` retained zero-filled tensor buffers in VRAM instead of immediate deallocation via `set_to_none=True`.
  6. `pipeline/09_streaming_eval.py`: Inconsistent passing of `ResolvedPaths` object instead of directory path to `should_skip_run` and `save_run_result`.
  7. Shared GPU VRAM threshold: `min_free_vram_gb` default lowered to 6.0 GB in `src/hdg/device.py` to prevent false aborts on shared nodes where free VRAM is ~8.74-8.76 GB.
  8. Codebase Linter Sweep: Systematically removed all unused imports, unused local variables, and static placeholder-free f-strings across `pipeline/`, `src/`, `tools/`, and `tests/`.
- **Patch**: All issues resolved and verified with `pyflakes` (0 undefined names, 0 syntax errors, 0 unused imports/variables) and full test suite (168 passed).

---

## 5. End-to-End Pipeline Stages & Real GPU Status

| Stage | Script | Purpose | Status on ISRO A100 |
| :--- | :--- | :--- | :--- |
| **0a** | `tools/check_system.sh` | Host & GPU hardware audit | **PASSED** (`results/real_gpu/env_manifest.json`) |
| **0b** | `tools/verify_gpu_graph.py` | Graph VRAM & memory cap check | **PASSED** |
| **0c** | `tools/verify_device_parity.py` | CPU vs GPU numerical parity | **PASSED** ($\lvert\Delta\rvert = 0.00210 \le 0.005$) |
| **5** | `pipeline.03_run_baselines` | Multi-seed baselines (TF-IDF, XGBoost, MLP) | **PASSED** (`results/real_gpu/tables/baselines_results.json`) |
| **6a** | `pipeline.04_train_gnn` | GraphSAGE & HGTConv Random Split (all 5 seeds) | **PASSED** (All 5 seeds for both SAGE & ATTN complete) |
| **6b** | `pipeline.04_train_gnn` | Temporal Split (both variants, all seeds) | **PASSED TRAINING** (Resuming to emit report & paired tests) |
| **6c** | `pipeline.04_train_gnn` | Group Split (component holdout) | Ready for sequential execution |
| **6d** | `pipeline.04b_stop_rules` | Pre-committed stop rules | Ready for sequential execution |
| **7** | `pipeline.05_ablation` | Topological & feature ablations | Ready for sequential execution |
| **8** | `pipeline.06_attack_eval` | Structural attacks & GNNGuard | Ready for sequential execution |
| **9** | `pipeline.07_latency_eval` | Latency benchmark (CPU-only) | Ready for sequential execution |
| **10** | `pipeline.08_provenance_checks`| Provenance & CESNET transfer | Ready for sequential execution |
| **11** | `pipeline.09_streaming_eval` | Continual streaming evaluation | Ready for sequential execution |
| **12** | `pipeline.10_emit_rpz` | BIND RPZ zone file emission | Ready for sequential execution |
| **13** | `pipeline.11_make_report` | Compile final `REPORT.md` | Ready for sequential execution |

---

## 6. How to Run on Remote Cluster

To resume the pipeline on the ISRO node:

```bash
cd /home/rohit/Malicious-HDG-self
git fetch origin main && git reset --hard origin/main
git log -1 --oneline
# Verify commit is 93dfcea (or latest main)
# Message: fix(pipeline): resolve undefined names, typing errors, and memory hygiene across all stages

bash pipeline/run_real_gpu.sh --device cuda
```

The runner will verify pre-flight checks, recognize that Stage 5, Stage 6a (both SAGE & ATTN), and Stage 6b models are already complete, and immediately continue through Stage 6c, 6d, 7, etc.
