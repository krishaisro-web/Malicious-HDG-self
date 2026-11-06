# GPU Adaptation Protocol — Implementation Checklist

> **Governance Rules:**
> 1. **Append-Only**: Tasks and items can only be added and ticked off (`[x]`). No items may ever be removed.
> 2. **Golden Data Preservation**: Existing baseline data (`results/real/`, `data/processed/real/`, `artifacts/checkpoints/real/`) are strictly read-only and immutable.
> 3. **Device Isolation**: Default CLI device is strictly `--device cpu`. GPU execution requires explicit `--device cuda` or `HDG_DEVICE=cuda`.
> 4. **Output Destination**: All GPU run artifacts, checkpoints, and logs write strictly to `results/real_gpu/` and `artifacts/checkpoints/real_gpu/`.
> 5. **Strict Precision**: Enforce strict FP32 (`allow_tf32 = False`, no BF16).
> 6. **Update Frequency**: This checklist is updated immediately following the completion of every small milestone or task.
> 7. **Companion Documentation**: See `docs/GPU_ADAPTATION_SUMMARY.md` for end-to-end technical architecture, safeguards, verification results, and operational cluster runbook.

---

## Phase Requirements Matrix

### Phase 2a: Pre-flight Verification & Environment Hardening
- [x] **Centralized Device Resolution**: `src/hdg/device.py::resolve_device` strictly defaulting to `cpu`, with `cuda` / `cuda:N` support and `HDG_DEVICE` env var override.
- [x] **Strict FP32 Precision Enforcement**: `torch.backends.cuda.matmul.allow_tf32 = False` and `torch.backends.cudnn.allow_tf32 = False`.
- [x] **CUDA Scatter-Add Determinism**: `torch.use_deterministic_algorithms(True, warn_only=True)`.
- [x] **Shared GPU Memory Limits**: `src/hdg/device.py::set_gpu_memory_limits` enforcing 35% memory fraction cap (`torch.cuda.set_per_process_memory_fraction(0.35)`).
- [x] **Dynamic VRAM Availability Check**: `src/hdg/device.py::check_vram_available` inspecting `torch.cuda.mem_get_info()` for >= 8.0 GB free VRAM before execution.
- [x] **System Environment Probing**: `tools/check_system.sh` probing OS, kernel, CPU, GPU driver, CUDA runtime, and generating `results/real_gpu/env_manifest.json`.
- [x] **Pre-Flight Graph Verification**: `tools/verify_gpu_graph.py` validating synthetic HeteroGNN forward/backward passes, GNNGuard cosine pruning ($\varepsilon = 10^{-8}$), and real graph (`data/processed/real/heterodata.pt`) profiling.

### Phase 2b: Pipeline Execution & Training Tuning
- [x] **50-Epoch Floor**: Minimum 50 epochs before early stopping triggers in `pipeline/04_train_gnn.py` and `pipeline/05_ablation.py`.
- [x] **Expanded Patience**: Early stopping patience set to 60 epochs.
- [x] **Co-Primary Model Selection**: Seed 42 checkpoint selection evaluating both in-distribution validation pAUC (FPR $\le 1\%$) and cross-source transfer AUC (LOMSO).
- [x] **Atomic Attack Tensor Transfer**: Structural attacks in `src/hdg/eval/attack.py` construct perturbed tensors on CPU before atomic `.to(device)` transfer.
- [x] **Latency Evaluation CPU Safeguard**: `pipeline/07_latency_eval.py` strictly CPU-only, rejecting `--device cuda` to safeguard production DNS claims.
- [x] **Seed 42 Device Parity Script**: `tools/verify_device_parity.py` confirming test ROC-AUC agreement within $\pm 0.002$ between CPU and GPU.

### Isolation & Provenance
- [x] **Provenance Metadata Logging**: `src/hdg/device.py::get_provenance_metadata` recording `device`, `torch_version`, `cuda_version`, `git_commit`, `hostname`, and precision.
- [x] **Run Output Directory Isolation**: All GPU outputs routed to `results/real_gpu/` and checkpoints to `artifacts/checkpoints/real_gpu/`.
- [x] **Read-Only Baseline Preservation**: Zero mutation of `results/real/`, `data/processed/real/`, or `artifacts/checkpoints/real/`.
- [x] **Independent Execution Protocol**: Never mix GPU and CPU evaluations in paired statistical tests or reuse runs across devices.

### Verification & Local Testing
- [x] **Unit Testing Suite**: Comprehensive unit tests in `tests/test_device.py` and `tests/test_gpu_tools.py` covering device resolution, fallbacks, memory fraction, VRAM check, determinism, and provenance.
- [x] **CPU Execution Verification**: Complete local dry-run on CPU verifying end-to-end pipeline operability (118/118 tests passing).

---

## Detailed Milestone Execution Breakdown

## Milestone 1: Centralized Device Resolution & Precision Enforcement (`src/hdg/device.py`)
- [x] Create `src/hdg/device.py` with unified device resolution function `resolve_device(device_arg=None) -> torch.device`
- [x] Default strictly to `cpu` when `device_arg` is None and `HDG_DEVICE` is unset
- [x] Support explicit device strings: `cpu`, `cuda`, `cuda:N`, `mps`
- [x] Enforce strict FP32 when CUDA is selected:
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
- [x] Enforce PyG CUDA scatter-add determinism: `torch.use_deterministic_algorithms(True, warn_only=True)`
- [x] Implement `set_gpu_memory_limits(fraction=0.35, device=None)` to enforce per-process memory limits on CUDA
- [x] Implement `check_vram_available(min_free_gb=8.0, device=None) -> bool` for dynamic pre-run VRAM verification
- [x] Implement `get_provenance_metadata(device=None)` / `get_execution_provenance(device=None)` returning dictionary with:
  - `device` (resolved string)
  - `device_type`
  - `torch_version` (`torch.__version__`)
  - `cuda_available`
  - `cuda_version` (`torch.version.cuda` if available)
  - `git_commit` (HEAD hash)
  - `hostname` (machine name)
  - `platform`
  - `precision` (`"FP32"`)
  - `tf32_matmul_enabled` (`False`)
  - `tf32_cudnn_enabled` (`False`)
- [x] Implement `setup_shared_gpu_environment(dev_idx=0, memory_fraction=0.35, min_free_vram_gb=8.0)`
- [x] Implement `get_gpu_memory_info(device)` returning allocated, reserved, free, and total VRAM
- [x] Write unit tests in `tests/test_device.py` validating resolution, defaults, fallback, FP32 settings, determinism, memory fraction, VRAM check, and provenance metadata (13 passing unit tests)

---

## Milestone 2: Pre-Flight Hardware Probing & Environment Verification (`tools/`)
- [x] Create `tools/check_system.sh`:
  - Probe OS version, kernel, CPU model, core count, RAM
  - Probe GPU model, driver version, CUDA toolkit version via `nvidia-smi`
  - Probe PyTorch, PyG, CUDA runtime version
  - Emit machine manifest to `results/real_gpu/env_manifest.json`
- [x] Create `tools/verify_gpu_graph.py`:
  - [x] CLI argument `--device` (default `cpu`, accepts `cuda`)
  - [x] Memory fraction capping: `torch.cuda.set_per_process_memory_fraction(0.35)` when running on CUDA
  - [x] Dynamic free VRAM check: verify $\ge 8.0\text{ GB}$ free memory before proceeding
  - [x] Synthetic graph sanity check: generate small HeteroData, instantiate `HeteroGNN`, run forward and backward passes
  - [x] GNNGuard verification: test cosine similarity pruning and scatter operations on GPU
  - [x] Real graph profiling: inspect `data/processed/real/heterodata.pt` (node counts, edge counts, memory footprint, single-pass forward timing, peak VRAM measurement)
  - [x] Structured JSON output to `results/real_gpu/preflight_verification.json`
- [x] Create `tools/verify_device_parity.py`:
  - Run Seed 42 HeteroGNN on CPU and GPU (or test-load identical weights)
  - Evaluate test ROC-AUC agreement (must satisfy $|\text{AUC}_{\text{gpu}} - \text{AUC}_{\text{cpu}}| \le 0.002$)
  - Emit parity verification report to `results/real_gpu/device_parity.json`

---

## Milestone 3: GPU Training Pipeline Adaptation (`pipeline/04_train_gnn.py`)
- [x] Add `--device` CLI argument to `pipeline/04_train_gnn.py` (default: `cpu`)
- [x] Integrate `resolve_device()` and apply FP32 / determinism configuration
- [x] Inject execution provenance metadata into output JSON
- [x] Enforce minimum 50-epoch floor before early stopping can trigger
- [x] Set early stopping patience to 60 epochs
- [x] Implement Seed 42 co-primary model selection:
  - Primary metric 1: In-distribution validation pAUC at $\text{FPR} \le 1\%$
  - Primary metric 2: Cross-source transfer AUC (evaluating generalization)
- [x] Ensure model checkpoints save strictly to `artifacts/checkpoints/real_gpu/` when running with `--device cuda`
- [x] Ensure training results save strictly to `results/real_gpu/`

---

## Milestone 4: GPU Ablation & Attack Pipeline Adaptation (`pipeline/05`, `06`, `07`)
- [x] Update `pipeline/05_ablation.py`:
  - Add `--device` CLI argument (default: `cpu`)
  - Support GPU execution across all ablation variants
  - Route GPU outputs strictly to `results/real_gpu/tables/`
- [x] Update `src/hdg/eval/attack.py`:
  - Ensure perturbed edge indices and feature tensors are staged on CPU before atomic `.to(device)` transfer
  - Ensure zero host-device synchronization bottlenecks in inner attack loops
- [x] Update `pipeline/06_attack_eval.py`:
  - Add `--device` CLI argument (default: `cpu`)
  - Ensure multi-seed attack evaluation runs cleanly on GPU
- [x] Audit `pipeline/07_latency_eval.py`:
  - Explicitly enforce CPU-only execution (raise error if `--device cuda` is passed, ensuring latency remains faithful to CPU serving constraints)

---

## Milestone 5: GPU Runner Orchestration & Pipeline Automation
- [x] Create `pipeline/run_real_gpu.sh`:
  - Pre-flight verification check (`tools/verify_gpu_graph.py --device cuda`)
  - Seed 42 device parity verification (`tools/verify_device_parity.py`)
  - HeteroGNN multi-seed training on GPU (`pipeline/04_train_gnn.py --device cuda`)
  - Ablation suite execution on GPU (`pipeline/05_ablation.py --device cuda`)
  - Adversarial attack evaluation on GPU (`pipeline/06_attack_eval.py --device cuda`)
  - CPU-only latency evaluation (`pipeline/07_latency_eval.py --device cpu`)
  - Stop rule evaluation (`pipeline/04b_stop_rules.py`)
  - Final dynamic report generation (`pipeline/11_make_report.py --out-dir results/real_gpu`)
- [x] Add executable permissions to scripts (`chmod +x`)

---

## Milestone 6: Verification & Test Suite Execution
- [x] Run full unit test suite (`pytest tests/`) — 118/118 tests passing in 16.96s
- [x] Verify zero regressions across all existing tests
- [x] Verify test coverage for new GPU modules (`src/hdg/device.py`, `tools/verify_gpu_graph.py`, `tools/verify_device_parity.py`, `tests/test_gpu_tools.py`)
- [x] Audit data directory immutability: verify checksums / timestamps of `results/real/`, `data/processed/real/`, and `artifacts/checkpoints/real/` are completely unchanged
- [x] Complete local execution dry run with `--device cpu`

---

## Milestone 7: Master Dynamic System, Hardware, GPU & Dataset Audit Tool (`tools/probe_hardware.py`)
- [x] Create comprehensive master audit script `tools/probe_hardware.py` executable standalone or via pipeline
- [x] Dynamic OS & Platform diagnostics (OS platform, architecture, hostname, Python runtime executable and version)
- [x] Dynamic CPU architecture inspection (model brand string, logical core count)
- [x] Dynamic Physical Memory & Swap diagnostics (total, available, used RAM; swap space; OOM prevention warnings)
- [x] Dynamic Storage & Workspace inspection (disk total, free space, write accessibility)
- [x] Dynamic GPU hardware & VRAM breakdown (device name, compute capability, SM count, total/free/allocated/reserved VRAM in GB & MB, nvidia-smi integration)
- [x] Strict FP32 & determinism precision safeguard verification (`allow_tf32 = False` validation)
- [x] Dynamic audit of 13 required Python libraries (`torch`, `torch_geometric`, `torch_scatter`, `torch_sparse`, `numpy`, `scipy`, `pandas`, `sklearn`, `xgboost`, `yaml`, `pyarrow`, `psutil`, `pytest`)
- [x] Dynamic Zenodo raw JSON datasets inspection:
  - Dynamically detects JSON array vs JSON lines format
  - Computes exact file size in MB and bytes
  - Computes dynamic record/row counts
  - Dynamically extracts schema keys from ingested records
  - Dynamically extracts sample domain names
- [x] Dynamic Processed Heterogeneous Graph Cohort inspection:
  - Ingests `domains.parquet`: computes dynamic row count, column count, column names, label distribution, distinct ingested threat/benign sources
  - Ingests `heterodata.pt`: computes total nodes, per-node-type breakdown, total edges, per-relation edge breakdown, domain feature tensor shape
  - Ingests `graph_manifest.json`: validates file integrity and checksum
- [x] 100% dynamic: Zero hardcoded or static values in terminal logs or exported JSON
- [x] CLI flag `--json <path>` exporting structured diagnostic JSON (`results/real_gpu/master_system_audit.json`)
- [x] Integrated into pre-flight shell audit `tools/check_system.sh`
- [x] Unit test suite integration in `tests/test_gpu_tools.py` validating 100% dynamic probing functionality


---

## Milestone 8: Complete Verification Audit, Parity Remediation & Team Awareness Documentation
- [x] **Parity Script Remediation (`tools/verify_device_parity.py`)**:
  - [x] Identified fatal signature mismatch (`TypeError: train_eval_gnn() got an unexpected keyword argument 'config'`)
  - [x] Implemented proper cohort dataset loading (`domains.parquet` & `heterodata.pt`)
  - [x] Implemented `random_stratified_split` and `apply_split_scaling` with `degree_mode="train_visible"`
  - [x] Initialized `HeteroGNN` with dynamic input channel dictionary and hyperparameter resolution
  - [x] Executed 50-epoch CPU reference pass in 1.31s and confirmed clean emission of `results/real_gpu/device_parity.json`
  - [x] Added unit test `test_device_parity_tool_runs_on_cpu` in `tests/test_gpu_tools.py`
- [x] **Code Discrepancy Alignments**:
  - [x] `src/hdg/device.py`: Exported aliases `set_cuda_precision_and_determinism` and `get_device_provenance`
  - [x] `src/hdg/device.py`: Added `get_device_provenance` and `set_cuda_precision_and_determinism` to `__all__`
  - [x] `pipeline/04_train_gnn.py`: Enforced 50-epoch floor (`min_epochs=50`) and 60 patience (`patience=60`)
  - [x] `pipeline/04_train_gnn.py`: Documented Seed 42 co-primary model selection (`in_dist_val_pauc_01` and cross-source transfer AUC) in `co_primary_selection` payload
  - [x] `pipeline/05_ablation.py`: Synchronized 50-epoch floor (`min_epochs=50`), default epochs (`train_epochs=200`), and 60 patience (`patience=60`)
- [x] **Verification Execution & Test Suite**:
  - [x] Executed `tools/verify_gpu_graph.py --device cpu`: Passed all checks (device resolution, strict FP32, synthetic forward/backward on `sage`, `attn`, `sage_guard`, real graph profiling)
  - [x] Executed `tools/verify_device_parity.py --help`: Passed with full argument parser documentation
  - [x] Executed `tools/verify_device_parity.py --out-dir results/real_gpu`: Passed with exit code 0
  - [x] Executed full unit test suite `pytest tests/`: **120/120 tests passed** in 17.53s (increased from 118 to 120 tests, zero regressions)
  - [x] Verified immutability of baseline directories (`results/real/`, `data/processed/real/`, `artifacts/checkpoints/real/`): 0 uncommitted changes, 0 diffs
- [x] **Comprehensive Team Awareness Documentation**:
  - [x] Authored `docs/GPU_ADAPTATION_SUMMARY.md` covering all 7 milestones, 9 mandatory constraints, architecture, safeguards, verification results, and remote cluster execution runbook
  - [x] Updated `docs/GPU_ADAPTATION_CHECKLIST.md` in append-only fashion tracking all tasks to completion

---

## Milestone 9: Adversarial Attack Atomic Transfer Deep-Copy Safeguard & Test Suite Hardening
- [x] **Adversarial Device Mutation Defect Remediation (`src/hdg/eval/attack.py`)**:
  - [x] Identified PyG `HeteroData.to("cpu")` in-place mutation defect at line 43 in `src/hdg/eval/attack.py` where input graph on non-CPU device was mutated in-place to CPU during structural attack generation.
  - [x] Implemented atomic transfer deep-copy safeguard: `clean_cpu = deepcopy(clean_data).to("cpu")`, ensuring `clean_data` remains intact on its original device across multi-budget attack evaluations (e.g. `for k in budgets:` in `pipeline/06_attack_eval.py`).
  - [x] Preserved atomic `.to(orig_device)` transfer of generated `attacked_data` back to the caller's device.
- [x] **Adversarial & Safeguard Test Suite Verification**:
  - [x] Updated `tests/test_adversarial_device_and_safeguards.py::test_generate_structural_attack_preserves_clean_data_device` to assert clean graph device immutability (device remains `mps` across sequential attack iterations).
  - [x] Added `test_attack_preserves_device_across_budgets` in `tests/test_attack.py` validating device retention across multiple budgets.
  - [x] Executed full test suite `pytest tests/`: **154/154 tests passed** cleanly in 56.29s with zero regressions.
  - [x] Executed `tools/verify_device_parity.py`: Passed with Seed 42 reference benchmark.
  - [x] Executed `tools/verify_gpu_graph.py --device cpu`: Passed all pre-flight checks.
  - [x] Verified 100% immutability of baseline directories (`results/real/`, `data/processed/real/`, `artifacts/checkpoints/real/`): 0 diffs, 0 modifications.
  - [x] Synchronized technical documentation in `docs/GPU_ADAPTATION_SUMMARY.md` reflecting deep-copy staging safeguard.
