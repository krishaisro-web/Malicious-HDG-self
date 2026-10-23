# Pipeline Root Cause Analysis & Engineering Resolution

This document details the root cause analysis, evidence, architectural defects, and definitive resolutions implemented for the Malicious-HDG pipeline across Root Causes A through J.

---

## 1. Root Cause A: Conflation of Single Chronological Holdout with Rolling-Origin Evaluation

### Defect
Previously, any temporal split was governed by a strict requirement that at least two individual calendar months had to contain both classes simultaneously. When evaluated on the Zenodo DomainRadar v2 dataset:
- Dec 2022 & Feb 2023 contained benign domains.
- Jul 2023 & Nov 2023 contained malware domains.
- Only Dec 2023 contained both benign and malware domains simultaneously (1 multi-class month).

Because `overlapping_months < 2`, the pipeline raised:
`ValueError: Overlapping months with both classes is 1 (< 2). Temporal rolling-origin evaluation cannot proceed.`
Furthermore, `pipeline/04_train_gnn.py` erroneously loaded `domains_timematched.parquet` for temporal splits, which discarded the preceding single-class historical training windows.

### Resolution
1. **Explicit Conceptual Separation**:
   - **Single Chronological Holdout**: A scientifically valid out-of-time evaluation where the historical training window comprises all domains strictly preceding target month $T_{\text{target}}$ ($t < T_{\text{target}}$). The training history is evaluated in *aggregate*: individual historical months are permitted to contain single classes, provided the aggregate historical training pool possesses adequate positive and negative class counts.
   - **Rolling-Origin Temporal Evaluation**: Evaluates sequential multi-window transitions ($T_k \to T_{k+1}$) with advancing training cutoffs. This mode strictly requires $\ge 2$ multi-class months and is reported separately.
2. **Implementation**:
   - `single_chronological_holdout_split()` in `src/hdg/data/temporal.py` implements the historical prefix holdout ($\text{History} \to \text{Dec 2023}$).
   - `rolling_origin_temporal_splits()` accepts `allow_empty=True` and produces empty transitions gracefully when multi-window conditions are not met.
   - Temporal evaluation always consumes the complete historical cohort (`domains.parquet`), retaining historical prefixes.

---

## 2. Root Cause B: Fragmented & Inconsistent Temporal Diagnostics Contract

### Defect
Different stages had conflicting definitions of temporal eligibility. The dataset profiler checked for multi-month overlap, the parser applied its own heuristic timestamp preference, and the downstream training script made independent assumptions. Furthermore, missing timestamps were at risk of being silently backfilled or misattributed.

### Resolution
1. **Unified Temporal Contract**:
   - Implemented `analyze_temporal_eligibility()` in `src/hdg/data/temporal.py`.
   - Produces a standardized dataclass `TemporalEligibilityResult` serialized to `results/<profile>/profile/temporal_eligibility.json`.
2. **Transparent Diagnostic Artifacts**:
   - Reports record counts per source, field availability coverage percentages (`sourced_on`, `evaluated_on`, `t_reg`, `t`), minimum/maximum months, per-source monthly distributions, and per-class monthly distributions.
   - Categorizes status into:
     - `SINGLE_HOLDOUT_ONLY`: Aggregate history has both classes, target month has both classes, rolling-origin unavailable.
     - `ROLLING_ORIGIN_AVAILABLE`: $\ge 2$ multi-class transitions available.
     - `BLOCKED_INSUFFICIENT_DATA`: Insufficient positive or negative samples in target or aggregate history.

---

## 3. Root Cause C: Information Leakage in Temporal Graph Snapshots

### Defect
Constructing a single static heterogeneous graph with future domains and infrastructure nodes and then partitioning nodes into historical train and future test sets creates severe transductive leakage:
- Shared infrastructure nodes (IPs, nameservers, ASNs, certificates) accumulate edges from future malware domains.
- Normalization statistics (degree features, log-transforms) incorporate future observations.
- Message-passing propagates representations backwards in time from future target domains into training domains.

### Resolution
1. **Temporal Graph Snapshot Pairing**:
   - Implemented `build_temporal_graph_pair(df, train_cutoff_month, target_month)` in `src/hdg/data/graph.py`.
   - **Training Snapshot ($G_{\text{train}}$)**: Strictly filtered to records with $t \le T_{\text{cutoff}}$. Future edges, domains, and incident infrastructure observed only in the future are strictly excluded.
   - **Evaluation Snapshot ($G_{\text{eval}}$)**: Includes target domains ($t = T_{\text{target}}$) and permits connections only to infrastructure active and observed up to the prediction window.
2. **Regression Testing**:
   - Added `test_temporal_cutoff_excludes_future_observations()` in `tests/test_temporal_diagnostics.py`, verifying that infrastructure nodes and edges introduced in future periods cannot alter the training snapshot node count or adjacency.

---

## 4. Root Cause D: Dataset–Graph Row and Index Misalignment

### Defect
`build_base_graph()` subsamples domains via `df.sample(max_domains).reset_index(drop=True)`. Downstream baseline and evaluation scripts independently re-read `domains.parquet` or applied independent random sampling. If the graph domain node ordering diverged from the DataFrame row indices, node feature tensors, split index arrays, and ground-truth labels became misaligned, corrupting evaluation metrics.

### Resolution
1. **Immutable Aligned Cohort Bundling**:
   - Implemented `save_graph_cohort_bundle()` in `src/hdg/data/graph.py`.
   - Whenever a graph is constructed or subsampled, it atomically writes:
     1. `heterodata.pt`: The PyTorch Geometric HeteroData artifact.
     2. `domains_cohort.parquet`: The exact, ordered DataFrame corresponding row-for-row with domain node index $0 \dots N-1$.
     3. `graph_manifest.json`: Cryptographic SHA-256 fingerprint, domain node count, relational edge counts, and schema version.
2. **Validated Cohort Loader**:
   - Implemented `load_aligned_graph_cohort()` used across all pipeline stages (`03_run_baselines`, `04_train_gnn`, `05_ablation`, `06_attack_eval`, `07_latency_eval`, `08_provenance_checks`, `09_streaming_eval`, `10_emit_rpz`).
   - Automatically asserts `len(df) == data['domain'].num_nodes` and `data['domain'].y == df['label']`.

---

## 5. Root Cause E: Inadequate Isolation Between Real, Smoke, and Fixture Profiles

### Defect
Passing `--smoke` created result subdirectories in some stages, but underlying processed data or checkpoints still defaulted to production paths (`data/processed/`, `artifacts/checkpoints/`). A quick `--smoke` run or `--limit 100` run risked permanently overwriting full production datasets and trained model weights.

### Resolution
1. **Three Completely Isolated Execution Profiles**:
   - `real`: `data/processed/real/`, `results/real/`, `artifacts/checkpoints/real/`
   - `smoke`: `data/processed/smoke/`, `results/smoke/`, `artifacts/checkpoints/smoke/`
   - `fixture`: `data/processed/fixture/`, `results/fixture/`, `artifacts/checkpoints/fixture/`
2. **Runtime Path Safeguards**:
   - `ResolvedPaths.check_write_path()` intercepts all write operations. If a smoke or fixture run attempts to write to a path containing production tokens, a `PermissionError` is raised immediately.
   - `pipeline/01_parse.py` enforces that supplying `--limit` without `--fixture` automatically routes outputs to the `smoke` profile.

---

## 6. Root Cause F: Naive File-Existence Caching and Resumption Fragility

### Defect
`should_skip_run()` previously inspected only whether an output JSON file existed and had a non-zero size. Modifying hyperparameters, changing model architectures, updating seed lists, or switching datasets was ignored because stale files triggered immediate skips. Checkpoint resumption did not verify if the checkpoint matched the current model configuration or dataset.

### Resolution
1. **Cryptographic Experiment Fingerprinting**:
   - Implemented in `src/hdg/cache.py`:
     - `compute_file_sha256()`: Chunked file hashing.
     - `compute_dict_fingerprint()`: Deterministic JSON canonicalization.
     - `compute_experiment_fingerprint()`: Hashes research configuration, input artifact hashes, split protocol, architecture, and random seeds.
   - `is_run_cache_valid()` validates the fingerprint stored inside run JSONs against the expected experiment fingerprint.
2. **Atomic Resumption & Checkpoint Verification**:
   - `train_eval_gnn()` embeds the `experiment_fingerprint` in epoch-level checkpoints. Resumed training rejects checkpoints with mismatched fingerprints and starts clean without corrupting the existing checkpoint.
   - Implemented `atomic_save_torch()` and `atomic_save_json()` via `tempfile.NamedTemporaryFile` + `fsync` + `os.replace` preventing partial-write file corruption during sudden interruption or power loss.

---

## 7. Root Cause G: Pipeline Runner Flag and Orchestration Inconsistencies

### Defect
`pipeline/run_all.py` did not propagate `--config` or `--force` to several stages (`00_profile_dataset.py`, `00b_audit_shortcuts.py`, `01_parse.py`, `02_build_graph.py`), leading to unrecognized argument crashes. `pipeline/run_real.sh` claimed that random, temporal, and group evaluations ran even when only random splits were invoked.

### Resolution
1. **Uniform CLI Interface**:
   - All pipeline stages (00 through 11) now implement a uniform argument contract: `--config`, `--force`, `--fixture`, `--smoke`, `--single-seed`, and `--max-domains`.
2. **Explicit Multi-Split Execution**:
   - `pipeline/run_all.py` and `pipeline/run_real.sh` explicitly iterate through all three scientific split protocols (`random`, `time`, `group`), reporting each split mode's metrics or honest blocked status.
3. **OpenMP Environment Thread Safety**:
   - `src/hdg/config.py` safely configures OpenMP, BLAS, and MKL runtime thread allocations upon initialization to prevent library initialization deadlocks on multi-threaded CPU environments.

---

## 8. Root Cause H: Crashing on Ineligible Research Experiments

### Defect
`pipeline/09_streaming_eval.py` raised unhandled exceptions when fewer than two multi-class months were available, abruptly terminating the entire pipeline run and leaving subsequent downstream reports ungenerated.

### Resolution
1. **Honest `BLOCKED_INSUFFICIENT_DATA` Status**:
   - When temporal prerequisites for continual streaming evaluation are not met, the stage outputs a machine-readable JSON:
     ```json
     {
       "experiment": "streaming_evaluation",
       "status": "BLOCKED_INSUFFICIENT_DATA",
       "reason": "Streaming continual-learning evaluation requires >= 2 consecutive multi-class months...",
       "available_months": ["2023-12"]
     }
     ```
   - Exits cleanly with status 0, allowing independent pipeline stages (RPZ emission, final reporting) to complete successfully.

---

## 9. Root Cause I: Untracked Checkpoint Selection in RPZ Generation

### Defect
`pipeline/10_emit_rpz.py` selected the first `.pt` file returned by an arbitrary filesystem glob without verifying which experiment, split protocol, or configuration produced it.

### Resolution
1. **Manifest-Linked Model Selection**:
   - Inspects `<model>_manifest.json` emitted alongside each trained model.
   - Verifies the model's experiment fingerprint, validation threshold calibration, and false-positive rate specifications before emitting RPZ response policy zones.
   - If no verified production model exists, it emits a `BLOCKED_MISSING_MODEL` artifact and refuses to emit speculative blocking rules.

---

## 10. Root Cause J: Conflation of Scientific Configuration with Hardware Configuration

### Defect
Dynamic hardware detection logic altered parameters affecting scientific metrics (such as dataset caps, model hidden dimensions, training epochs, and random seed counts) between different machines, making cross-machine comparisons invalid.

### Resolution
1. **Decoupled Configuration Hierarchy**:
   - **Research Configuration**: Defines scientific methodology—model architecture, layer counts, features, loss functions, learning rate schedules, seeds, split protocols, and evaluation metrics.
   - **Runtime Configuration**: Controls hardware resource allocation—CPU thread caps (`HDG_THREADS`), batch sizes for memory constraints, device settings (`cpu` vs `cuda`), and I/O buffer sizes.
2. **Hardware-Invariant Fingerprints**:
   - `compute_experiment_fingerprint()` explicitly filters out runtime keys (`hardware`, `runtime`, `threads`, `device`, `paths`), ensuring identical scientific experiments produce identical fingerprints on both development machines and the ISRO HP Thin Client.
