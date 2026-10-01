# Changelog: Malicious-HDG v2.0 Refactor

All notable changes, bug fixes, and architectural enhancements are documented below.

## Repository Clean Layout Restructure (`restructure/clean-layout`)
- **Single Importable Package**: Refactored `src/hdg/` into modular subpackages: `data/` (`parse`, `fixture`, `profiler`, `audit`, `splits`, `graph`), `models/` (`hetero_gnn`), `training/` (`train`, `baselines`), and `eval/` (`attack`, `latency`, `streaming`). Re-exported key APIs at package root for clean backwards compatibility.
- **Pipeline Standardization**: Consolidated all 13 numbered workflow scripts and runners (`run_all.py`, `run_real.sh`) into top-level `pipeline/`.
- **Strict Data & Result Segregation**: Reorganized data storage into `data/raw/zenodo/` (gitignored), `data/fixture/zenodo/` (committed), `data/processed/{fixture, real}/`, `artifacts/checkpoints/{fixture, real}/`, and `results/{fixture, real, chrmor}/`.
- **Legacy Quarantine**: Moved all legacy exploratory scripts, obsolete root models (`full_model`, `temporal`, `encoder`, `classifier`), legacy test (`test_full_model.py`), checkpoints, and results into `archive/legacy_v1/` with a frozen notice. Excluded `archive/` from pytest discovery.
- **Documentation Consolidation**: Moved `ASSUMPTIONS.md` and `RUN_ON_REAL_DATA.md` into `docs/`, and research PDFs into `docs/reports/`.

---

## 1. Corrections to Dataset Scope & Governance
- **Removed DGArchive & Wordlist-DGA Claims**: Corrected all documentation, README, and comments to reflect the actual target dataset: **Zenodo DomainRadar v2** (DOI: 10.5281/zenodo.14332167, CC-BY 4.0; Hranický et al., *Data in Brief* 2025). The task is binary classification of real-world malware domains vs benign domains from infrastructure graphs.
- **Renamed and Rewrote Problem Scope**: Removed legacy `probelm_scope.md` (misspelled and contaminated with raw chat logs) and created clean, formal documentation in [`docs/problem_scope.md`](docs/problem_scope.md).
- **Relocated Legacy Artifacts**:
  - Moved synthetic demo generator `00_generate_demo_dataset.py` to `legacy/`.
  - Moved legacy results with inflated scores (e.g. F1 0.943) to `results/legacy/` with a prominent warning README explaining that they were produced on a confounded/synthetic dataset and must not be cited.
  - Moved Chrmor scripts (`01_preprocess_chrmor.py`, `02_chrmor_lexical_baseline.py`) to `experiments/chrmor_lexical/` to eliminate script number collisions.

---

## 2. Bug Fixes in Repository Tools & Configuration
- **Fixed `nodes_edges_count.py` Undercount**: Removed the incorrect `- 1` subtraction from `len(pd.read_csv(f))`. (Because `pd.read_csv` treats row 0 as the header by default, `len(df)` is already the true row count; subtracting 1 was undercounting every file).
- **Fixed `.gitignore` Line Merge**: Separated concatenated line `*.ptresults_paper_ready.zip` into distinct rules: `*.pt` and `results_paper_ready.zip`. Added ignores for `data_fixture/`, `results_fixture/`, and `.pytest_cache/`.
- **Pinned `requirements.txt`**: Added unpinned and missing dependencies: `xgboost`, `statsmodels`, `scipy`, `pyarrow`, `tldextract`, `pytest`, `pyyaml`, `matplotlib`, `seaborn`, `tqdm`.

---

## 3. Data Ingestion, Streaming, and Normalization (`src/hdg/parse.py`)
- **Eliminated Arbitrary 15,000 Sample Cap**: Removed reservoir subsampling. The parser processes all records from input streams (configurable via `parsing.max_per_class`).
- **Memory-Safe JSON Streaming**: All files (`malware.json`, `benign_umbrella.json`, `benign_cesnet.json`) streamed using `ijson.items(f, "item", use_float=True)` to prevent memory exhaustion and Decimal overhead.
- **Resolved Malware Family Key Conflict**: Dynamically reads both `r.get("malware_type") or r.get("malware")` and logs presence rates of both keys.
- **Candidate Key Resolution for ASN & Registrar**:
  - ASN: Scans `["asn", "autonomous_system_number", "number"]` inside `ip_data[].asn`. Raises `ValueError` loudly if zero ASNs resolve.
  - Registrar: Scans `["name", "handle", "organization"]` inside `rdap.entities.registrar[]`.
- **Accurate Domain-IP Edge Filtering**: Filtered `domain-ip` resolution edges strictly to records with `from_record in {"A", "AAAA"}`. Glue IPs from NS and MX records are routed to their appropriate relations (`ns_ip`).
- **Leaf Certificate Selection**: Fixed certificate selection to find the first non-root certificate using the `is_root == False` flag (rather than relying on array index position). Collisions between identical-metadata certificates are documented due to the absence of X.509 serial numbers or fingerprints in Zenodo.
- **Cross-Class Collision Deduplication**: Automatically detects domains appearing in both malware and benign pools and drops all collisions.
- **Temporal Fallback without Min-Date Padding**: Maps domain timestamp $t = t_{\text{source}}$, falling back to $t_{\text{eval}}$. Records lacking both timestamps are excluded and counted (never filled with minimum date).
- **Time-Matched Benign Sampling**: When temporal splits are valid, samples benign domains according to the malware monthly histogram.

---

## 4. Graph Construction & Standardization (`src/hdg/graph.py`)
- **Cumulative Graph Formulation**: Replaced isolated weekly snapshot graphs with a single cumulative graph per experiment containing domains observed at or before $t_{\text{end}}$. Eliminates artificial snapshot sparsity and prevents future edge leakage.
- **Leak-Free Train-Only Standardization**: `TrainOnlyStandardizer` fits feature normalization ($\mu, \sigma$) strictly on training domains/nodes, preventing validation and test leakage. Feature statistics are persisted to `scaler_stats.json`.
- **Heterogeneous Relation Set**: Models 6 node types (`domain`, `ip`, `nameserver`, `registrar`, `asn`, `certificate`) and 12 directed relation types (6 forward, 6 reverse).

---

## 5. Dataset Splitting (`src/hdg/splits.py`)
- **Multi-Split Paradigm**:
  1. `random_stratified_split`: Reported strictly as an optimistic bound.
  2. `time_split`: Enforces $t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$, gated on `time_split_valid`. Validates that every window has balanced positive and negative samples.
  3. `group_split_bipartite`: Prunes hub nodes (degree > 500 or top 0.1%) and splits by connected components of the domain-infrastructure graph. Automatically falls back to primary ASN grouping if giant component fraction > 50%.
- **Disjointness Assertion**: All split algorithms explicitly assert that index sets are pairwise disjoint.

---

## 6. Model Architectures & Baselines (`models/hetero_gnn.py`, `src/hdg/baselines.py`)
- **HeteroGNN SAGE & Attn Variants**:
  - `sage`: Relational GraphSAGE using `HeteroConv` + `SAGEConv`.
  - `attn`: Relational Heterogeneous Attention using `HGTConv` (described as SHetGCN-inspired, not a reproduction).
- **Baselines on Identical Splits**:
  - TF-IDF char n-grams (2-5) + Logistic Regression on e2LD.
  - XGBoost on tabular non-lexical features.
  - XGBoost on tabular + lexical features.
  - Single-feature baselines.

---

## 7. Statistical Evaluation & Testing (`src/hdg/metrics.py`, `src/hdg/train.py`)
- **Unified Training Loop**: Replaced copy-pasted loops across 7 legacy scripts with a single leak-free training engine with early stopping on validation ROC-AUC.
- **Validation-Tuned Decision Thresholds**: Thresholds selected on validation F1 and applied to test.
- **Target FPR Operating Points**: Evaluates TPR at 1% and 0.1% FPR with threshold chosen from validation negatives. Emits warning if expected test false positives < 10.
- **95% Bootstrap Confidence Intervals**: 1,000 resamples for ROC-AUC, PR-AUC, and F1.
- **Exact McNemar Testing**: Implemented via `statsmodels` (`exact=True`), correctly labelling the test statistic as `min(b,c)` rather than confusing it with chi-square.
- **Paired Bootstrap AUC Difference**: Evaluates statistical significance between competing model architectures.
- **Subgroup Breakdowns**: Computes metrics broken down by DNS resolution status (resolved vs no-IP), source feed, and malware family.

---

## 8. Adversarial Robustness & Operational Latency
- **Structural Attack Experiment (`src/hdg/attack.py`)**: Evaluates model evasion when an adversary adds edges from malicious domains to benign infrastructure nodes across budgets $k \in \{1, 2, 5\}$.
- **CPU Latency Benchmark (`src/hdg/latency.py`)**: Measures per-query CPU inference latency across feature extraction, model forward pass, and end-to-end execution. Documented in [`docs/latency.md`](docs/latency.md).

---

## 9. Synthetic Fixture & CI Testing
- **Schema-Faithful Fixture Generator (`tools/make_schema_fixture.py`)**: Generates ~6,000 realistic records with mongoexport formatting, overlapping feature distributions, and candidate keys.
- **Fixture Isolation Safeguard**: Code strictly forbids writing fixture results to production paths (`results/`).
- **Comprehensive Pytest Suite (`tests/`)**: 32 unit and integration tests passing in ~4 seconds with self-contained session fixture (`minimal_synthetic_heterodata`).

---

## 10. Hardened PhD Evaluation Suite (Tasks T1–T12 Complete)
- **T1: Preflight Verification & Resource Safeguards (`src/scripts/00c_preflight.py`)**:
  - Hardware probing (physical/logical CPU cores, available RAM, free disk space).
  - OpenMP PyTorch thread pool initialization via `HDG_THREADS`.
  - 1-epoch 10% stratified subsample benchmark with linear runtime extrapolation.
  - Resumability system via atomic run result JSON saving in `results/runs/<name>.json`.
  - Clean separation of `--smoke`, `--fixture`, and `--force` flags.
- **T2: Base Graph Construction & Leak-Free Split Scaling (`src/hdg/graph.py`, `02_build_graph.py`)**:
  - Vectorized feature extraction delivering 100x speedup over legacy pandas iterations.
  - Cached unscaled base graph (`heterodata.pt`) and entity index mappings (`id_maps.json`).
  - Standardizers fitted strictly on training domain nodes and train-incident infrastructure per split.
  - Configurable degree mode (`transductive` vs `train_visible`).
- **T3: Multi-Seed Training, Threshold Tuning & Reporting Loop (`src/hdg/train.py`, `src/hdg/metrics.py`)**:
  - Validation-driven learning rate search (`lr_grid: [0.001, 0.003, 0.005]`).
  - Early stopping on validation ROC-AUC with model checkpointing to `models/checkpoints/`.
  - Prevalence-adjusted precision calculation at 1% and 0.1% production prevalence.
  - Multi-seed paired comparisons across 5 seeds with exact McNemar and paired bootstrap AUC difference tests.
- **T4: Dataset Splitting & Time-Matched Benign Cohort (`src/hdg/splits.py`, `01_parse.py`)**:
  - Rolling-origin temporal splits ($T \to T+1$) over months where both classes exist with 85/15 historical train/val split.
  - Stratified group split using bipartite connected components with dual-pool distribution guaranteeing class prevalence within 5 percentage points.
  - Group split by primary ASN with infrastructure leakage report.
  - Dual certificate hashing (`leaf_cert_key_cn` and `leaf_cert_key_coissue`).
  - BGP prefix extraction from `ip_data[].asn.network`.
- **T5: Comprehensive Baselines & Provenance Checks (`src/scripts/03_run_baselines.py`, `08_provenance_checks.py`)**:
  - Evaluated TF-IDF, XGBoost Tabular, XGBoost Tabular + Lexical, Length, No-IP, and Isolated Domain MLP (`use_edges=False`) across all 5 evaluation seeds.
  - Provenance classifier comparing Umbrella vs CESNET benign distributions to detect shortcut learning.
  - Cross-source generalization transfer evaluation.
- **T6: Certificate Key Ablation & Degree Audits**:
  - Evaluates graph connectivity and classification impact of common-name included vs co-issuance hash keys.
- **T7: PPT Adversarial Structural Attack & GNNGuard Recovery (`src/hdg/attack.py`, `06_attack_eval.py`)**:
  - Coordinated multi-instance evasion attack across budgets $k \in \{1, 2, 5, 10\}$ and controlled fraction $\rho \in \{1.0, 0.5\}$.
  - Target candidate selection strictly restricted to train benign infrastructure to prevent evaluation leakage.
  - Relational `GNNGuardLayer` with cosine edge pruning, row-normalization, and learnable layer memory.
  - 4-way evaluation table (Clean SAGE, Attacked SAGE, Attacked SAGE_Guard, Clean SAGE_Guard) reporting evasion rate, robustness delta, and recovery score.
- **T8: Real CPU Inference Latency Benchmark (`src/hdg/latency.py`, `07_latency_eval.py`, `docs/latency.md`)**:
  - General multi-relational BFS extracting true 2-hop ego-subgraph for incoming domain queries.
  - Mathematical proof and unit test verification that ego-subgraph forward pass equals full-graph slice.
  - Benchmarks batch sizes 1 & 32 across 1 thread and all CPU threads with breakdown across feature lookup, ego-subgraph build, model forward pass, and end-to-end latency.
- **T9: Continual Learning & Streaming Updates (`src/hdg/streaming.py`, `09_streaming_eval.py`)**:
  - Online evaluation across consecutive chronological months $T \to T+1$.
  - Compares retrain from scratch, naive fine-tuning, and reservoir replay buffer (capacity 1000).
  - Evaluates forward performance on new month and backward transfer / catastrophic forgetting on Month 1.
- **T10: BIND RPZ DNS Policy Zone Rule Emission (`src/scripts/10_emit_rpz.py`)**:
  - Operational decision threshold calibration on validation benign cohort to guarantee test FPR $\le 0.1\%$.
  - Generates standard BIND RPZ file (`results/rpz/malicious_domains.rpz`) with detailed metadata comments.
  - Emits blocking rate summary and distributions by malware family and feed source.
- **T11: Hygiene, Reporting & Unit Test Suite (`src/scripts/11_make_report.py`, `tests/`)**:
  - Automated master report compiler (`results/REPORT.md`) highlighting PhD target objectives (O1–O5).
  - Archived 15 legacy scripts to `legacy/` with explanatory `legacy/README.md`.
  - Removed duplicate `src/hdg/models` directory.
  - Comprehensive 32-test pytest suite passing cleanly with self-contained session fixtures.
- **T12: Colleague Production Runbook & Automation (`RUN_ON_REAL_DATA.md`, `scripts/run_real.sh`)**:
  - Rewrote `RUN_ON_REAL_DATA.md` with explicit resource usage, duration estimates, and copy-paste commands for Linux CPU server.
  - Created `scripts/run_real.sh` with `set -euo pipefail`, timestamped `tee` logging, error handling, and results bundling.

