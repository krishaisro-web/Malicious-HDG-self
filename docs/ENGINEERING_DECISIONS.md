# Engineering & Scientific Methodology Decisions

This document records key scientific findings, external literature citations, and architectural decisions made while engineering the Malicious-HDG pipeline.

---

## 1. DomainRadar v2 Dataset & Timestamp Semantics

### Primary Source Findings
- **Dataset Reference**: Zenodo Record [10.5281/zenodo.10629548](https://zenodo.org/records/10629548) (*DomainRadar: A Large-scale, Longitudinal Dataset for Malicious Domain Detection*).
- **Observed Timestamp Fields**:
  - `sourced_on`: ISO-8601 UTC timestamp indicating when the domain was ingested or reported by threat intelligence feeds (URLhaus, ThreatFox, OpenPhish) or benign top-lists.
  - `evaluated_on`: Timestamp when active feature extraction (DNS queries, TLS handshakes, RDAP lookup) was performed by the collection scanner.
  - `rdap.registration_date`: Parsed WHOIS/RDAP creation date ($t_{\text{reg}}$).
  - Effective time $t$: Derived observation timestamp.

### Methodological Policy
1. **Ingestion Preference**: `sourced_on` is preferred as the ground-truth observation cutoff date because it reflects when the domain emerged in the wild. If absent, `evaluated_on` is accepted as an empirical fallback.
2. **Missing Timestamps**: Missing dates are never arbitrarily imputed with epoch zero or synthetic timestamps; records without valid timestamps are excluded from chronological splits and flagged in `temporal_eligibility.json`.
3. **Historical Single-Class Windows**: In threat intelligence feeds, certain feeds report batches of active attacks in specific months (e.g., July 2023, November 2023), while benign baseline sampling occurs in others (December 2022, February 2023). Rejecting historical training pools solely because individual months are not perfectly balanced is scientifically unmotivated; an aggregate historical window that provides sufficient positive and negative domains represents a defensible training pool for temporal holdout evaluation.

---

## 2. Chronological Validation & Information Leakage Prevention

### Authoritative Literature & Principles
- **Arp et al. (2022)**: *Dos and Don'ts of Machine Learning in Computer Security* (USENIX Security 2022). Emphasizes that future-information leakage and transductive graph connections across time boundaries lead to drastically inflated and invalid performance estimates.
- **Pendlebury et al. (2019)**: *TESSERACT: Eliminating Experimental Bias in Malware Classification across Space and Time* (USENIX Security 2019). Demonstrates that temporal evaluations must strictly constrain training-time representations to records preceding the prediction period.

### Engineering Decisions in Malicious-HDG
1. **Strictly Train-Only Standardization**:
   - Implemented `TrainOnlyStandardizer` in `src/hdg/data/scaler.py`.
   - Feature means ($\mu$) and standard deviations ($\sigma$) for continuous numerical features (domain lexical, TTL, cert duration) and infrastructure node degree counts are computed strictly on training domains ($\mathcal{D}_{\text{train}}$) and train-incident infrastructure nodes ($\mathcal{I}_{\text{train}}$).
   - Test and validation representations receive identical scaling parameters without updating them.
2. **Snapshot-Based Temporal Separation**:
   - `build_temporal_graph_pair()` partitions topology chronologically. Edges formed after training cutoff $T$ are omitted from the training base graph $G_{\text{train}}$. Future infrastructure nodes appearing only in target period $U$ do not exist in $G_{\text{train}}$.
   - Future labels $\mathcal{Y}_{\text{eval}}$ are strictly withheld from graph construction, threshold calibration, and hyperparameter selection.

---

## 3. Heterogeneous Graph Construction & PyTorch Geometric

### Official PyTorch Geometric Guidance
- HeteroData maintains per-node-type feature matrices ($X \in \mathbb{R}^{N_v \times D_v}$) and per-relation edge index matrices ($E \in \mathbb{R}^{2 \times M_e}$).
- Relational message passing in `HeteroConv` and `HGTConv` requires contiguous integer indexing $0 \dots N_v-1$ per node type.

### Implemented Index Alignment Contract
1. **Graph Manifest**: Each graph saved to disk is accompanied by `domains_cohort.parquet` and `graph_manifest.json`.
2. **Strict Cohort Binding**: Downstream evaluation scripts cannot independently sample or alter `domains.parquet`. They must load the cohort via `load_aligned_graph_cohort()`, guaranteeing that row index $i$ in the DataFrame corresponds identically to domain node index $i$ in `data['domain'].x` and `data['domain'].y`.

---

## 4. Checkpoint Resumption & Atomic Serialization

### PyTorch Official Guidelines
- Checkpointing for long-running training jobs must save not only `model.state_dict()` but also `optimizer.state_dict()`, `scheduler.state_dict()`, current epoch, best validation metrics, and early-stopping counters to ensure exact state restoration upon resumption.

### Implemented Resumption Contract
1. **Fingerprint Invalidation**: Checkpoints store the cryptographic `experiment_fingerprint`. Resumption verifies that the stored fingerprint matches the current experiment configuration. If hyperparameters, split protocols, or architectures have changed, the stale checkpoint is rejected, and training begins fresh under an isolated identifier.
2. **Atomic Writes**: Both model checkpoints (`.pt`) and run metadata (`.json`) are serialized using `NamedTemporaryFile` + `fsync` + `os.replace`, guaranteeing that sudden termination cannot leave corrupted partial files on disk.

---

## 5. Clean Separation of Research vs. Runtime Hardware Configurations

### Architectural Rule
- **Research Parameters**: Randomness seeds, model layers, hidden dimensions, attention heads, dropout rates, learning rates, epochs, patience, feature sets, split protocols, and evaluation thresholds define the scientific contribution. These must remain identical across all machines to guarantee comparability.
- **Runtime Parameters**: CPU thread allocations (`OMP_NUM_THREADS`, `torch.set_num_threads`), memory safety limits, device placement (`cpu` vs `cuda`), and file buffer sizes adapt to machine hardware without altering research parameters.
- **Fingerprint Verification**: `compute_experiment_fingerprint()` hashes research configuration while stripping runtime parameters, ensuring that experiments executed on developer workstations and ISRO CPU hardware share identical research fingerprints.
