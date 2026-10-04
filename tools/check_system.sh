#!/usr/bin/env bash
# ==============================================================================
# Malicious-HDG Hardware & Environment Diagnostics for ISRO HP Thin Client
# Run this command on your ISRO workstation terminal:
#     bash tools/check_system.sh
# You can copy-paste the entire output into ChatGPT!
# ==============================================================================

echo "======================================================================"
echo "    ISRO HP THIN CLIENT HARDWARE & ENVIRONMENT AUDIT"
echo "======================================================================"

# 1. OS & Kernel
echo ""
echo "[1] OPERATING SYSTEM & KERNEL:"
if [ -f /etc/os-release ]; then
    grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"'
else
    uname -s -r -m
fi
uname -a

# 2. CPU Architecture & Cores
echo ""
echo "[2] CPU MODEL & CORES:"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\):|CPU max MHz" || true
else
    grep "model name" /proc/cpuinfo | head -n 1 || true
    echo "Total logical cores: $(nproc 2>/dev/null || echo 'Unknown')"
fi
CPU_CORES=$(nproc 2>/dev/null || echo 1)
echo "Logical Threads Detected (nproc): ${CPU_CORES}"

# 3. RAM & Swap Space
echo ""
echo "[3] MEMORY & SWAP (CRITICAL FOR LINUX OOM KILLER PREVENTION):"
free -h || true
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
TOTAL_RAM_GB=$(( TOTAL_RAM_KB / 1024 / 1024 ))
SWAP_TOTAL_KB=$(grep SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo 0)
SWAP_TOTAL_GB=$(( SWAP_TOTAL_KB / 1024 / 1024 ))

echo "  -> Physical RAM: ${TOTAL_RAM_GB} GB"
echo "  -> Swap Space:   ${SWAP_TOTAL_GB} GB"

if [ "$SWAP_TOTAL_GB" -lt 2 ]; then
    echo ""
    echo "  [!] WARNING: Low or zero swap detected (${SWAP_TOTAL_GB} GB)."
    echo "  PyTorch in-memory graph allocation may get killed by Linux OOM killer."
    echo "  Run this to create a 4GB swapfile:"
    echo "    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
fi

# 4. Storage Space
echo ""
echo "[4] AVAILABLE DISK SPACE IN WORKSPACE:"
df -h . | awk 'NR==1 || NR==2'

# 5. Python & PyTorch Environment
echo ""
echo "[5] PYTHON & PYTORCH ENVIRONMENT:"
if command -v python3 >/dev/null 2>&1; then
    python3 --version
    python3 -c "import torch; print(f'PyTorch {torch.__version__}, CPU Threads: {torch.get_num_threads()}, CUDA Available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not yet installed in current environment."
else
    echo "python3 not found in PATH."
fi

# 6. Raw Data Directory Check
echo ""
echo "[6] RAW ZENODO DATASET FILES CHECK (data/raw/zenodo):"
for f in malware.json benign_umbrella.json benign_cesnet.json; do
    if [ -f "data/raw/zenodo/$f" ]; then
        SIZE=$(ls -lh "data/raw/zenodo/$f" | awk '{print $5}')
        echo "  [FOUND] data/raw/zenodo/$f ($SIZE)"
    else
        echo "  [MISSING] data/raw/zenodo/$f"
    fi
# 7. Dynamic Threading & RAM Rationale Calculation
REC_THREADS=2
THREAD_REASON="Default safe threading"
if [ "$CPU_CORES" -le 2 ]; then
    REC_THREADS=$CPU_CORES
    THREAD_REASON="System has ${CPU_CORES} core(s). Use all available core(s)."
elif [ "$CPU_CORES" -le 4 ]; then
    REC_THREADS=2
    THREAD_REASON="System has ${CPU_CORES} cores. Capping to 2 threads prevents thermal throttling on fanless/low-TDP chassis and leaves headroom for OS."
elif [ "$CPU_CORES" -le 8 ]; then
    REC_THREADS=$(( CPU_CORES - 1 > 4 ? 4 : CPU_CORES - 1 ))
    THREAD_REASON="System has ${CPU_CORES} cores. Using ${REC_THREADS} threads for optimal PyTorch parallel efficiency without context thrashing."
else
    HALF=$(( CPU_CORES / 2 ))
    REC_THREADS=$(( HALF > 8 ? 8 : (HALF < 4 ? 4 : HALF) ))
    THREAD_REASON="System has ${CPU_CORES} cores. Using ${REC_THREADS} threads (higher thread counts on CPU GNNs exhibit diminishing returns and high memory per thread)."
fi

echo ""
echo "======================================================================"
echo "      CHATGPT-READY DYNAMIC CALIBRATION FOR configs/hp_thin_client.yaml"
echo "======================================================================"
echo "Copy-paste this exact diagnostic block into ChatGPT, or apply directly to"
echo "configs/hp_thin_client.yaml:"
echo ""
echo "Hardware Detected: ${CPU_CORES} CPU Cores | ${TOTAL_RAM_GB} GB RAM | ${SWAP_TOTAL_GB} GB Swap"
echo "Threading Rationale: ${THREAD_REASON}"
echo ""
echo "--- RECOMMENDED YAML SNIPPET ---"
echo "hardware:"
echo "  num_threads: ${REC_THREADS}  # ${THREAD_REASON}"

if [ "$TOTAL_RAM_GB" -le 4 ]; then
    echo "graph:"
    echo "  max_domains: 15000"
    echo "parsing:"
    echo "  max_per_class: 7500"
    echo "model:"
    echo "  sage:"
    echo "    hidden_dim: 16"
    echo "    out_dim: 8"
    echo "  attn:"
    echo "    hidden_dim: 16"
    echo "    out_dim: 8"
    echo "  training:"
    echo "    epochs: 40"
    echo "    batch_size: 128"
    echo "randomness:"
    echo "  evaluation_seeds: [42, 1337]"
elif [ "$TOTAL_RAM_GB" -le 8 ]; then
    echo "graph:"
    echo "  max_domains: 30000"
    echo "parsing:"
    echo "  max_per_class: 15000"
    echo "model:"
    echo "  sage:"
    echo "    hidden_dim: 32"
    echo "    out_dim: 16"
    echo "  attn:"
    echo "    hidden_dim: 32"
    echo "    out_dim: 16"
    echo "  training:"
    echo "    epochs: 50"
    echo "    batch_size: 256"
    echo "randomness:"
    echo "  evaluation_seeds: [42, 1337, 2024]"
elif [ "$TOTAL_RAM_GB" -le 16 ]; then
    echo "graph:"
    echo "  max_domains: 50000"
    echo "parsing:"
    echo "  max_per_class: 25000"
    echo "model:"
    echo "  sage:"
    echo "    hidden_dim: 48"
    echo "    out_dim: 24"
    echo "  attn:"
    echo "    hidden_dim: 48"
    echo "    out_dim: 24"
    echo "  training:"
    echo "    epochs: 60"
    echo "    batch_size: 512"
    echo "randomness:"
    echo "  evaluation_seeds: [42, 1337, 2024]"
else
    echo "======================================================================"
    echo "  [ENTERPRISE SERVER DETECTED] (~${TOTAL_RAM_GB} GB RAM, ${CPU_CORES} CORES)"
    echo "======================================================================"
    echo "Your server has ~${TOTAL_RAM_GB} GB RAM! You have immense memory headroom."
    echo "You DO NOT need configs/hp_thin_client.yaml or subsampling restrictions."
    echo "Use the standard production configuration: configs/default.yaml"
    echo "Run the full pipeline with:"
    echo "  bash pipeline/run_real.sh"
    echo "======================================================================"
fi


