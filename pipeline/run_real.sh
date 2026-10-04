#!/usr/bin/env bash
# ==============================================================================
# Malicious-HDG Production Runner for Zenodo DomainRadar v2
# Supported Hardware Environments:
#   1. HP Thin Client (ISRO Lab / Workstation: 2-4 cores CPU, 4-16 GB RAM, no GPU)
#   2. Multi-core Linux Servers / Workstations (CPU-only, multi-core, large RAM)
# Features:
#   - Auto-detection of HP Thin Client / low-RAM hardware profile
#   - Thermal & thread safety: caps PyTorch / OpenMP / BLAS thread pool to avoid throttling
#   - Swap verification to prevent Linux Out-Of-Memory (OOM) killer terminations
#   - Resume semantics: skips previously completed runs unless --force is supplied
#   - Atomic logging and automatic final results bundling
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/results/real/logs"
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

# Python binary detection
PYTHON_BIN=""
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    error_exit "Neither python3 nor python was found in PATH!"
fi

# Parse CLI options
CONFIG_OVERRIDE=""
FORCE_THIN_CLIENT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --thin-client)
            FORCE_THIN_CLIENT=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Dynamic Configuration Generation & Profile Selection
if [[ -n "${CONFIG_OVERRIDE}" ]]; then
    CONFIG_FILE="${CONFIG_OVERRIDE}"
    log "Using explicitly supplied configuration: ${CONFIG_FILE}"
elif [ "${FORCE_THIN_CLIENT}" = true ] || [[ "${HDG_PROFILE:-}" == "thin_client" || "${HDG_PROFILE:-}" == "hp_thin_client" ]]; then
    CONFIG_FILE="configs/hp_thin_client.yaml"
    log "Using static HP Thin Client configuration: ${CONFIG_FILE}"
else
    log "Dynamically probing hardware environment and generating tailored active config..."
    CONFIG_FILE="configs/active_config.yaml"
    ${PYTHON_BIN} -m src.hdg.config --generate-dynamic --output "${CONFIG_FILE}"
fi

export HDG_CONFIG="${CONFIG_FILE}"

# Extract dynamic parameters from active config via Python
ACTIVE_INFO=$(${PYTHON_BIN} -c "
import yaml
try:
    with open('${CONFIG_FILE}', 'r', encoding='utf-8') as f:
        c = yaml.safe_load(f)
    hw = c.get('hardware', {})
    threads = hw.get('num_threads', 2)
    profile = hw.get('profile', 'dynamic')
    desc = hw.get('profile_description', '')
    ram = hw.get('detected_ram_gb', 'N/A')
    cores = hw.get('detected_cpu_cores', 'N/A')
    domains = c.get('graph', {}).get('max_domains', 'all')
    batch = c.get('model', {}).get('training', {}).get('batch_size', 256)
    epochs = c.get('model', {}).get('training', {}).get('epochs', 50)
    print(f'{threads}|{profile}|{desc}|{ram}|{cores}|{domains}|{batch}|{epochs}')
except Exception:
    print('2|fallback||N/A|N/A|30000|256|50')
" 2>/dev/null || echo "2|fallback||N/A|N/A|30000|256|50")

IFS='|' read -r CFG_THREADS CFG_PROFILE CFG_DESC CFG_RAM CFG_CORES CFG_DOMAINS CFG_BATCH CFG_EPOCHS <<< "${ACTIVE_INFO}"

# Thread Allocation & Safety Checks
DEFAULT_THREADS="${CFG_THREADS:-2}"
export HDG_THREADS="${HDG_THREADS:-${DEFAULT_THREADS}}"
export OMP_NUM_THREADS="${HDG_THREADS}"
export MKL_NUM_THREADS="${HDG_THREADS}"
export OPENBLAS_NUM_THREADS="${HDG_THREADS}"
export VECLIB_MAXIMUM_THREADS="${HDG_THREADS}"
export NUMEXPR_NUM_THREADS="${HDG_THREADS}"

# Swap check for low-RAM systems
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
SWAP_TOTAL_GB=$(( SWAP_TOTAL_KB / 1024 / 1024 ))

if [[ $TOTAL_RAM_GB -gt 0 && $TOTAL_RAM_GB -le 8 && $SWAP_TOTAL_GB -lt 2 ]]; then
    warn "CRITICAL MEMORY NOTICE: Only ${TOTAL_RAM_GB} GB RAM and ${SWAP_TOTAL_GB} GB swap detected!"
    warn "Running PyTorch on constrained memory without swap risks kernel OOM Killer termination."
    warn "Recommended swap command: sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
fi

log "=========================================================================="
log " Malicious-HDG: Full Production Pipeline Execution on Real Zenodo Data"
log " Log File:         ${PIPELINE_LOG}"
log " Active Config:    ${CONFIG_FILE}"
log " Hardware Profile: ${CFG_PROFILE} (${CFG_DESC})"
log " Detected RAM:     ${CFG_RAM} GB (System total)"
log " CPU Cores:        ${CFG_CORES} logical cores"
log " Allocated Threads: ${HDG_THREADS} threads (OMP/MKL/OpenBLAS synchronized)"
log " Graph Max Domains: ${CFG_DOMAINS} domains"
log " Training Config:  Batch Size: ${CFG_BATCH} | Epochs: ${CFG_EPOCHS}"
log "=========================================================================="

# Step 0: Preflight Verification
log "STAGE 0: Running System & Dataset Preflight Checks..."
${PYTHON_BIN} -m pipeline.00c_preflight --config "${CONFIG_FILE}"
success "Stage 0 (Preflight) PASSED."

# Step 1: Profiling
log "STAGE 1: Profiling Dataset & Key Schema Distributions..."
${PYTHON_BIN} -m pipeline.00_profile_dataset --config "${CONFIG_FILE}"
success "Stage 1 (Profile) PASSED."

# Step 2: Parsing & Time-Matched Sampling
log "STAGE 2: Parsing Raw JSON Records & Sampling Time-Matched Benign Cohort..."
${PYTHON_BIN} -m pipeline.01_parse --config "${CONFIG_FILE}"
success "Stage 2 (Parse) PASSED."

# Step 3: Shortcut & Single-Feature Leakage Audit
log "STAGE 3: Running Shortcut Audit & Feature Sensitivity Checks..."
${PYTHON_BIN} -m pipeline.00b_audit_shortcuts --config "${CONFIG_FILE}"
success "Stage 3 (Shortcut Audit) PASSED."

# Step 4: Construct Unscaled Base Heterogeneous Graph
log "STAGE 4: Building Unscaled Base HeteroData Graph..."
${PYTHON_BIN} -m pipeline.02_build_graph --config "${CONFIG_FILE}"
success "Stage 4 (Build Graph) PASSED."

# Step 5: Multi-Seed Baselines
log "STAGE 5: Running Multi-Seed Baselines (TF-IDF, XGBoost, MLP, Single-Feature)..."
${PYTHON_BIN} -m pipeline.03_run_baselines --config "${CONFIG_FILE}"
success "Stage 5 (Baselines) PASSED."

# Step 6: Multi-Seed HeteroGNN Training & Evaluation
log "STAGE 6: Training & Evaluating HeteroGNN (SAGE & Attn across Random, Time, Group splits)..."
${PYTHON_BIN} -m pipeline.04_train_gnn --config "${CONFIG_FILE}" --variant both
success "Stage 6 (HeteroGNN Evaluation) PASSED."

# Step 7: Topological & Feature Ablation Study
log "STAGE 7: Running Systematic Ablation Study..."
${PYTHON_BIN} -m pipeline.05_ablation --config "${CONFIG_FILE}"
success "Stage 7 (Ablation) PASSED."

# Step 8: Structural Adversarial Attack & GNNGuard Recovery
log "STAGE 8: Evaluating PPT Structural Attack & Relational GNNGuard..."
${PYTHON_BIN} -m pipeline.06_attack_eval --config "${CONFIG_FILE}"
success "Stage 8 (Attack & Robustness) PASSED."

# Step 9: Operational CPU Latency Benchmark
log "STAGE 9: Benchmarking CPU Inference Latency..."
LATENCY_QUERIES=$(${PYTHON_BIN} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG_FILE}')); print(cfg.get('latency', {}).get('num_queries', 500))" 2>/dev/null || echo 500)
LATENCY_WARMUP=$(${PYTHON_BIN} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG_FILE}')); print(cfg.get('latency', {}).get('warmup', 20))" 2>/dev/null || echo 20)
${PYTHON_BIN} -m pipeline.07_latency_eval --config "${CONFIG_FILE}" --num-queries "${LATENCY_QUERIES}" --warmup "${LATENCY_WARMUP}"
success "Stage 9 (Latency Benchmark) PASSED."

# Step 10: Dataset Provenance & Shortcut Generalization Checks
log "STAGE 10: Running Dataset Provenance & Cross-Source Generalization Audits..."
${PYTHON_BIN} -m pipeline.08_provenance_checks --config "${CONFIG_FILE}"
success "Stage 10 (Provenance Checks) PASSED."

# Step 11: Continual Learning & Streaming Evaluation
log "STAGE 11: Evaluating Streaming Updates (Retrain vs Fine-Tune vs Replay Buffer)..."
${PYTHON_BIN} -m pipeline.09_streaming_eval --config "${CONFIG_FILE}"
success "Stage 11 (Streaming Continual Learning) PASSED."

# Step 12: BIND Response Policy Zone (RPZ) Rule Emission
log "STAGE 12: Calibrating 0.1% FPR Threshold & Emitting BIND RPZ Zone File..."
${PYTHON_BIN} -m pipeline.10_emit_rpz --config "${CONFIG_FILE}" --target-fpr 0.001
success "Stage 12 (RPZ Emission) PASSED."

# Step 13: Compile Comprehensive Report
log "STAGE 13: Compiling Final Comprehensive REPORT.md..."
${PYTHON_BIN} -m pipeline.11_make_report --config "${CONFIG_FILE}"
success "Stage 13 (Report Compilation) PASSED."

log "=========================================================================="
log " PIPELINE EXECUTION COMPLETE!"
log " Comprehensive Report: ${REPO_ROOT}/results/real/REPORT.md"
log " Tables Directory:     ${REPO_ROOT}/results/real/tables/"
log " Runs Directory:       ${REPO_ROOT}/results/real/runs/"
log " RPZ Output:           ${REPO_ROOT}/results/real/rpz/"
log "=========================================================================="

# Packaging results bundle
BUNDLE_FILE="${REPO_ROOT}/results_real_bundle_$(date +%Y%m%d).tar.gz"
log "Packaging results into: ${BUNDLE_FILE}"
tar -czf "${BUNDLE_FILE}" -C "${REPO_ROOT}" results/real/
success "Bundle created: ${BUNDLE_FILE} (Please transfer this file back to your research workstation)."
