#!/usr/bin/env bash
# ==============================================================================
# Malicious-HDG Production GPU Runner for NVIDIA A100 (Phase 2 Protocol)
# Adheres to docs/GPU_ADAPTATION_PLAN.md & Phase 2 specifications.
#
# Constraints & Safeguards Enforced:
#   1. Protocol Duality: Independent execution protocol writing strictly to results/real_gpu/
#   2. Explicit Device: Defaults to --device cuda (never silent auto-switch)
#   3. Strict FP32: allow_tf32 = False, zero BF16/AMP
#   4. Run Isolation: Preserves results/real/ as read-only reference data
#   5. PyG CUDA scatter determinism enabled
#   6. Atomic CPU-to-GPU tensor staging in adversarial attacks
#   7. Latency Isolation: Stage 9 strictly runs on CPU (--device cpu)
#   8. Shared GPU Protection: 35% memory fraction cap (~28 GB) + dynamic VRAM verification (>= 8.0 GB)
#   9. Complete Evidence Pipeline Coverage: Pre-flight, Parity, GNNs, Ablations, Attacks, Latency, Stop Rules, Report
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_DEVICE="cuda"
CONFIG_OVERRIDE=""
FORCE_RERUN=false
SKIP_PREFLIGHT=false
EXTRA_PIPELINE_FLAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            GPU_DEVICE="$2"
            shift 2
            ;;
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --force)
            FORCE_RERUN=true
            EXTRA_PIPELINE_FLAGS+=("--force")
            shift
            ;;
        --skip-preflight)
            SKIP_PREFLIGHT=true
            shift
            ;;
        *)
            echo "Unrecognized argument: $1"
            echo "Usage: bash pipeline/run_real_gpu.sh [--device cuda|cuda:N] [--config <path>] [--force] [--skip-preflight]"
            exit 1
            ;;
    esac
done

export HDG_DEVICE="${GPU_DEVICE}"

# Dedicated output directories (Constraint #4)
OUT_DIR="${REPO_ROOT}/results/real_gpu"
LOG_DIR="${OUT_DIR}/logs"
CKPT_DIR="${REPO_ROOT}/artifacts/checkpoints/real_gpu"

mkdir -p "${LOG_DIR}" "${OUT_DIR}/tables" "${OUT_DIR}/runs" "${OUT_DIR}/profile" "${OUT_DIR}/rpz" "${CKPT_DIR}"
PIPELINE_LOG="${LOG_DIR}/pipeline_gpu_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log() {
    echo -e "\n\033[1;34m[$(date '+%Y-%m-%d %H:%M:%S')] $1\033[0m"
}

success() {
    echo -e "\033[1;32m[$(date '+%Y-%m-%d %H:%M:%S')] $1\033[0m"
}

error_exit() {
    echo -e "\033[1;31m[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m" >&2
    exit 1
}

trap 'error_exit "Pipeline aborted prematurely at line $LINENO!"' ERR

# Python binary detection
PYTHON_BIN="python3"
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
fi

# Configuration selection
if [[ -n "${CONFIG_OVERRIDE}" ]]; then
    CONFIG_FILE="${CONFIG_OVERRIDE}"
    log "Using explicitly supplied configuration: ${CONFIG_FILE}"
else
    CONFIG_FILE="configs/default.yaml"
    log "Using default production configuration: ${CONFIG_FILE}"
fi

export HDG_CONFIG="${CONFIG_FILE}"

log "=========================================================================="
log "   MALICIOUS-HDG A100 GPU EXECUTION PIPELINE (PHASE 2 PROTOCOL)"
log "=========================================================================="
log " Target Device:     ${GPU_DEVICE}"
log " Output Directory:  ${OUT_DIR}"
log " Checkpoints:       ${CKPT_DIR}"
log " Precision:         Strict FP32 (allow_tf32 = False, No BF16)"
log " Python Binary:     ${PYTHON_BIN}"
log " Reference Base:    results/real/ (Read-Only Immutable Reference)"
log "=========================================================================="

# ------------------------------------------------------------------------------
# STAGE 0: Pre-Flight Hardware Probing & Safeguard Verification
# ------------------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT}" = false ]; then
    log "STAGE 0a: Probing Host Environment & Emitting env_manifest.json..."
    bash tools/check_system.sh

    log "STAGE 0b: Verifying Pre-Flight GPU Graph Constraints & Shared Memory Cap..."
    ${PYTHON_BIN} tools/verify_gpu_graph.py --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" --mem-fraction 0.35 --min-free-vram 8.0

    log "STAGE 0c: Verifying Seed 42 Device Parity Tolerance (|AUC_gpu - AUC_cpu| <= 0.002)..."
    ${PYTHON_BIN} tools/verify_device_parity.py --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" --max-delta 0.002
    success "Pre-flight environment and hardware checks PASSED."
else
    log "Pre-flight checks skipped (--skip-preflight supplied)."
fi

# ------------------------------------------------------------------------------
# STAGE 5: Multi-Seed Baselines (TF-IDF, XGBoost, MLP, Single-Feature)
# ------------------------------------------------------------------------------
log "STAGE 5: Running Multi-Seed Baselines (identical splits)..."
${PYTHON_BIN} -m pipeline.03_run_baselines --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 5 (Baselines) PASSED."

# ------------------------------------------------------------------------------
# STAGE 6: Multi-Seed HeteroGNN Training & Evaluation on GPU (Random, Time, Group)
# ------------------------------------------------------------------------------
log "STAGE 6a: Training & Evaluating HeteroGNN on GPU (Random Split with Certified Calibration)..."
${PYTHON_BIN} -m pipeline.04_train_gnn --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" --split-type random --variant both ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}

log "STAGE 6b: Training & Evaluating HeteroGNN on GPU (Time Split - Inference-Only Temporal)..."
${PYTHON_BIN} -m pipeline.04_train_gnn --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" --split-type time --variant both ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}

log "STAGE 6c: Training & Evaluating HeteroGNN on GPU (Group Split - Infrastructure Component Holdout)..."
${PYTHON_BIN} -m pipeline.04_train_gnn --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" --split-type group --variant both ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}

log "STAGE 6d: Evaluating Pre-Committed Stop Rules (Post-Hoc Stacker & pAUC Fine-Tuning)..."
${PYTHON_BIN} -m pipeline.04b_stop_rules --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}/tables" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 6 (HeteroGNN Random, Time, Group, Stop Rules on GPU) PASSED."

# ------------------------------------------------------------------------------
# STAGE 7: Topological & Feature Ablation Study on GPU
# ------------------------------------------------------------------------------
log "STAGE 7: Running Systematic Ablation Study on GPU (Strict FP32)..."
${PYTHON_BIN} -m pipeline.05_ablation --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 7 (Ablation on GPU) PASSED."

# ------------------------------------------------------------------------------
# STAGE 8: Purity-Ranked Structural Adversarial Attack & Relational GNNGuard on GPU
# ------------------------------------------------------------------------------
log "STAGE 8: Evaluating Purity-Ranked Structural Attack & Relational GNNGuard on GPU..."
${PYTHON_BIN} -m pipeline.06_attack_eval --config "${CONFIG_FILE}" --device "${GPU_DEVICE}" --out-dir "${OUT_DIR}/tables" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 8 (Attack & Robustness Differences on GPU) PASSED."

# ------------------------------------------------------------------------------
# STAGE 9: Operational CPU Latency Benchmark & Sampling Pareto Frontier
# Constraint #7: Strictly CPU-only to maintain DNS production serving validity
# ------------------------------------------------------------------------------
log "STAGE 9: Benchmarking CPU Inference Latency & Sampling Pareto Frontier (Strictly CPU-Only)..."
LATENCY_QUERIES=$(${PYTHON_BIN} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG_FILE}')); print(cfg.get('latency', {}).get('num_queries', 500))" 2>/dev/null || echo 500)
LATENCY_WARMUP=$(${PYTHON_BIN} -c "import yaml; cfg=yaml.safe_load(open('${CONFIG_FILE}')); print(cfg.get('latency', {}).get('warmup', 20))" 2>/dev/null || echo 20)
${PYTHON_BIN} -m pipeline.07_latency_eval --config "${CONFIG_FILE}" --device cpu --num-queries "${LATENCY_QUERIES}" --warmup "${LATENCY_WARMUP}" --out-dir "${OUT_DIR}/tables" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 9 (CPU-Only Latency Pareto Benchmark) PASSED."

# ------------------------------------------------------------------------------
# STAGE 10: Dataset Provenance, CESNET Audit & Cross-Source Transfer Audits
# ------------------------------------------------------------------------------
log "STAGE 10: Running Dataset Provenance, CESNET Audit & Cross-Source Transfer Audits..."
${PYTHON_BIN} -m pipeline.08_provenance_checks --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}/profile" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 10 (Provenance & Cross-Source Transfer Checks) PASSED."

# ------------------------------------------------------------------------------
# STAGE 11: Continual Learning & Streaming Evaluation
# ------------------------------------------------------------------------------
log "STAGE 11: Evaluating Streaming Updates (Retrain vs Fine-Tune vs Replay Buffer)..."
${PYTHON_BIN} -m pipeline.09_streaming_eval --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 11 (Streaming Continual Learning) PASSED."

# ------------------------------------------------------------------------------
# STAGE 12: BIND Response Policy Zone (RPZ) Rule Emission
# ------------------------------------------------------------------------------
log "STAGE 12: Calibrating Certified 0.1% FPR Threshold & Emitting BIND RPZ Zone File..."
${PYTHON_BIN} -m pipeline.10_emit_rpz --config "${CONFIG_FILE}" --target-fpr 0.001 --out-dir "${OUT_DIR}" ${EXTRA_PIPELINE_FLAGS[@]+"${EXTRA_PIPELINE_FLAGS[@]}"}
success "Stage 12 (Certified RPZ Emission) PASSED."

# ------------------------------------------------------------------------------
# STAGE 13: Compile 100% Dynamic Final Comprehensive REPORT.md
# ------------------------------------------------------------------------------
log "STAGE 13: Compiling 100% Dynamic REPORT.md from GPU Run Artifacts..."
${PYTHON_BIN} -m pipeline.11_make_report --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}" --force
success "Stage 13 (Report Compilation) PASSED."

log "=========================================================================="
log " GPU PIPELINE EXECUTION COMPLETE!"
log " Comprehensive Report: ${OUT_DIR}/REPORT.md"
log " Tables Directory:     ${OUT_DIR}/tables/"
log " Runs Directory:       ${OUT_DIR}/runs/"
log " RPZ Output:           ${OUT_DIR}/rpz/"
log " Checkpoints:          ${CKPT_DIR}"
log "=========================================================================="

BUNDLE_FILE="${REPO_ROOT}/results_real_gpu_bundle_$(date +%Y%m%d).tar.gz"
log "Packaging GPU results into: ${BUNDLE_FILE}"
tar -czf "${BUNDLE_FILE}" -C "${REPO_ROOT}" results/real_gpu/
success "GPU Bundle created: ${BUNDLE_FILE}"
