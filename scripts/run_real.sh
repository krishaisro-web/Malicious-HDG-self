#!/usr/bin/env bash
# ==============================================================================
# Malicious-HDG Production Runner for Zenodo DomainRadar v2
# Target Server: CPU-only Linux Server (~245 GB RAM, Python 3.12, no GPU)
# Handles: resume/skip completed stages, error checking, tee logging, timestamps
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/results/logs"
mkdir -p "${LOG_DIR}"
PIPELINE_LOG="${LOG_DIR}/pipeline_real_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log() {
    echo -e "\n\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] $1\033[0m"
}

success() {
    echo -e "\033[1;32m[$(date '+%Y-%m-%d %H:%M:%S')] $1\033[0m"
}

warn() {
    echo -e "\033[1;33m[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1\033[0m"
}

error_exit() {
    echo -e "\033[1;31m[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m" >&2
    exit 1
}

# Trap unexpected exits
trap 'error_exit "Pipeline aborted prematurely at line $LINENO!"' ERR

log "=========================================================================="
log " Malicious-HDG: Full Production Pipeline Execution on Real Zenodo Data"
log " Log File: ${PIPELINE_LOG}"
log " CPU Cores Detected: $(nproc || echo 'Unknown')"
log " RAM Available: $(free -h | awk '/^Mem:/ {print $2}')"
log "=========================================================================="

export HDG_THREADS="${HDG_THREADS:-$(nproc)}"
log "PyTorch thread pool configured to: ${HDG_THREADS} threads"

# Step 0: Preflight Verification
log "STAGE 0: Running System & Dataset Preflight Checks..."
python -m src.scripts.00c_preflight
success "Stage 0 (Preflight) PASSED."

# Step 1: Profiling
log "STAGE 1: Profiling Dataset & Key Schema Distributions..."
python -m src.scripts.00_profile_dataset
success "Stage 1 (Profile) PASSED."

# Step 2: Parsing & Time-Matched Sampling
log "STAGE 2: Parsing Raw JSON Records & Sampling Time-Matched Benign Cohort..."
python -m src.scripts.01_parse
success "Stage 2 (Parse) PASSED."

# Step 3: Shortcut & Single-Feature Leakage Audit
log "STAGE 3: Running Shortcut Audit & Feature Sensitivity Checks..."
python -m src.scripts.00b_audit_shortcuts
success "Stage 3 (Shortcut Audit) PASSED."

# Step 4: Construct Unscaled Base Heterogeneous Graph
log "STAGE 4: Building Unscaled Base HeteroData Graph..."
python -m src.scripts.02_build_graph
success "Stage 4 (Build Graph) PASSED."

# Step 5: Multi-Seed Baselines
log "STAGE 5: Running Multi-Seed Baselines (TF-IDF, XGBoost, MLP, Single-Feature)..."
python -m src.scripts.03_run_baselines
success "Stage 5 (Baselines) PASSED."

# Step 6: Multi-Seed HeteroGNN Training & Evaluation
log "STAGE 6: Training & Evaluating HeteroGNN (SAGE & Attn across Random, Time, Group splits)..."
python -m src.scripts.04_train_gnn --variant both
success "Stage 6 (HeteroGNN Evaluation) PASSED."

# Step 7: Topological & Feature Ablation Study
log "STAGE 7: Running Systematic Ablation Study (35 Relation & Feature Variations)..."
python -m src.scripts.05_ablation
success "Stage 7 (Ablation) PASSED."

# Step 8: Structural Adversarial Attack & GNNGuard Recovery
log "STAGE 8: Evaluating PPT Structural Attack & Relational GNNGuard..."
python -m src.scripts.06_attack_eval
success "Stage 8 (Attack & Robustness) PASSED."

# Step 9: Operational CPU Latency Benchmark
log "STAGE 9: Benchmarking CPU Inference Latency (Batch sizes 1 & 32 across single & all threads)..."
python -m src.scripts.07_latency_eval --num-queries 500 --warmup 20
success "Stage 9 (Latency Benchmark) PASSED."

# Step 10: Dataset Provenance & Shortcut Generalization Checks
log "STAGE 10: Running Dataset Provenance & Cross-Source Generalization Audits..."
python -m src.scripts.08_provenance_checks
success "Stage 10 (Provenance Checks) PASSED."

# Step 11: Continual Learning & Streaming Evaluation
log "STAGE 11: Evaluating Streaming Updates (Retrain vs Fine-Tune vs Replay Buffer)..."
python -m src.scripts.09_streaming_eval
success "Stage 11 (Streaming Continual Learning) PASSED."

# Step 12: BIND Response Policy Zone (RPZ) Rule Emission
log "STAGE 12: Calibrating 0.1% FPR Threshold & Emitting BIND RPZ Zone File..."
python -m src.scripts.10_emit_rpz --target-fpr 0.001
success "Stage 12 (RPZ Emission) PASSED."

# Step 13: Compile Comprehensive Report
log "STAGE 13: Compiling Final Comprehensive REPORT.md..."
python -m src.scripts.11_make_report
success "Stage 13 (Report Compilation) PASSED."

log "=========================================================================="
log " PIPELINE EXECUTION COMPLETE!"
log " Comprehensive Report: ${REPO_ROOT}/results/REPORT.md"
log " Tables Directory:     ${REPO_ROOT}/results/tables/"
log " Runs Directory:       ${REPO_ROOT}/results/runs/"
log " RPZ Output:           ${REPO_ROOT}/results/rpz/"
log "=========================================================================="

# Packaging results bundle
BUNDLE_FILE="${REPO_ROOT}/results_bundle_$(date +%Y%m%d).tar.gz"
log "Packaging results into: ${BUNDLE_FILE}"
tar -czf "${BUNDLE_FILE}" results/REPORT.md results/tables/ results/runs/ results/rpz/ results/logs/
success "Bundle created: ${BUNDLE_FILE} (Please transfer this file back to your research workstation)."
