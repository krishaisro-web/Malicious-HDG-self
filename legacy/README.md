# Legacy Scripts Archive

This directory archives early prototype and development scripts that have been superseded by the hardened, leak-free Malicious-HDG pipeline (`src/scripts/` and `src/hdg/`).

## Superseded Scripts & Rationale

| Legacy Script | Superseded By | Reason for Supersession |
|:---|:---|:---|
| `00_check_dataset.py` | `src/scripts/00c_preflight.py` | Minimal 20-line raw file existence check replaced by thorough preflight validator, dependency verification, and resource checker. |
| `01_parse_zenodo_unified.py` | `src/scripts/01_parse.py` | Replaced by single-pass JSON streaming parser with BGP prefix extraction, dual certificate hashing, and time-matched benign sampling. |
| `02_build_node_tables.py` | `src/scripts/02_build_graph.py` | Replaced by vectorized feature extraction and unscaled base graph caching (`heterodata.pt`). |
| `03_build_edges_snapshots.py` | `src/scripts/02_build_graph.py` | Eliminated redundant snapshot edge tables; graph topology is now cached directly. |
| `04_build_heterodata.py` | `src/scripts/02_build_graph.py` | Global standardizers were previously fitted across entire graphs, causing test-set leakage. Now standardizers are fit strictly on training splits. |
| `05_train_dry_run.py` | `src/scripts/00c_preflight.py` | Integrated into preflight hardware, dependency, and throughput benchmarks. |
| `06_create_split.py` | `src/hdg/splits.py` | Replaced by leak-free split generators (rolling-origin temporal, stratified group, ASN leakage audit). |
| `07_cpu_sanity_check.py` | `src/scripts/00c_preflight.py` | Integrated into CPU core and OpenMP threading preflight checks. |
| `10_train.py` | `src/scripts/04_train_gnn.py` | Replaced by unified `train_eval_gnn` with early stopping, validation learning rate search, and checkpointing. |
| `11_evaluate.py` | `src/hdg/train.py`, `src/hdg/metrics.py` | Metrics calculation unified into `evaluate_predictions` with operating points and bootstrap CIs. |
| `12_ablation.py` | `src/scripts/05_ablation.py` | Vectorized in-memory evaluation of 35 topological and feature ablations without rebuilding raw tables. |
| `13_rolling_origin.py` | `src/scripts/04_train_gnn.py` | Rolling-origin temporal splits now evaluated natively within the unified training pipeline. |
| `14_temporal_holdout.py` | `src/scripts/04_train_gnn.py` | Integrated into unified split evaluation. |
| `17_check_window_18_19.py` | `src/scripts/08_provenance_checks.py` | Replaced by systematic dataset provenance and cross-source generalization audits. |
| `18_static_baseline_check.py` | `src/scripts/03_run_baselines.py` | Baselines (TF-IDF, XGBoost tabular/lexical, MLP no-edges, single-feature) unified into one multi-seed script. |
| `nodes_edges_count.py` | `src/scripts/00_profile_dataset.py` | Dataset topology and node/edge statistics now profiled systematically. |
