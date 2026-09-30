# Changelog: Malicious-HDG v2.0 Refactor

All notable changes, bug fixes, and architectural enhancements are documented below.

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
- **Comprehensive Pytest Suite (`tests/`)**: 17 tests verifying parsing edge cases, leaf certificate selection, cumulative graph properties, scaler isolation, split disjointness, and end-to-end integration.
