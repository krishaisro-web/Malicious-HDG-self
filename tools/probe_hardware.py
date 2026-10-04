#!/usr/bin/env python3
"""
Cross-platform hardware and environment probing tool for Malicious-HDG.
Outputs CPU, RAM, Swap, Disk, and PyTorch specs.
Run on your ISRO workstation:
    python3 tools/probe_hardware.py
You can copy-paste the entire output into ChatGPT!
"""

import os
import platform
import shutil
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    print("=" * 70)
    print("    ISRO HP THIN CLIENT HARDWARE & ENVIRONMENT AUDIT (PYTHON)")
    print("=" * 70)

    # 1. OS & Platform
    print("\n[1] OPERATING SYSTEM:")
    print(f"  OS:      {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python:  {sys.version.split()[0]} ({sys.executable})")

    # 2. CPU
    cpu_count = os.cpu_count() or 1
    print("\n[2] CPU & THREADS:")
    print(f"  CPU Logical Cores: {cpu_count}")

    # 3. RAM & Swap
    print("\n[3] MEMORY & SWAP:")
    ram_gb = 0.0
    swap_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        ram_gb = round(vm.total / (1024**3), 2)
        swap_gb = round(sm.total / (1024**3), 2)
        avail_ram = round(vm.available / (1024**3), 2)
        free_swap = round(sm.free / (1024**3), 2)
        print(f"  Physical RAM: {ram_gb} GB total ({avail_ram} GB available)")
        print(f"  Swap Space:   {swap_gb} GB total ({free_swap} GB free)")
    except Exception:
        # Fallback for Linux /proc/meminfo
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            mem_dict = {}
            with open(meminfo, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem_dict[parts[0].strip()] = parts[1].strip()
            tot_kb = int(mem_dict.get("MemTotal", "0 kB").split()[0])
            swap_kb = int(mem_dict.get("SwapTotal", "0 kB").split()[0])
            ram_gb = round(tot_kb / (1024**2), 2)
            swap_gb = round(swap_kb / (1024**2), 2)
            print(f"  Physical RAM: {ram_gb} GB total")
            print(f"  Swap Space:   {swap_gb} GB total")

    if swap_gb < 2.0 and ram_gb <= 8.5:
        print("\n  [!] WARNING: Low/Zero Swap Space detected!")
        print("  Running PyTorch on an 4GB-8GB thin client without swap can cause Linux OOM kill.")
        print("  Recommended command:")
        print("    sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")

    # 4. Disk Space
    print("\n[4] DISK SPACE IN REPO:")
    total, used, free = shutil.disk_usage(repo_root)
    print(f"  Workspace Free Disk: {round(free / (1024**3), 2)} GB (Total: {round(total / (1024**3), 2)} GB)")

    # 5. PyTorch
    print("\n[5] PYTORCH CHECK:")
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  PyTorch Threads: {torch.get_num_threads()}")
        print(f"  CUDA Available:  {torch.cuda.is_available()}")
    except ImportError:
        print("  PyTorch not installed in this environment.")

    # 6. Raw Data Files
    print("\n[6] ZENODO DATASET FILES (data/raw/zenodo/):")
    raw_dir = repo_root / "data" / "raw" / "zenodo"
    for fn in ["malware.json", "benign_umbrella.json", "benign_cesnet.json"]:
        p = raw_dir / fn
        if p.exists():
            mb = round(p.stat().st_size / (1024**2), 1)
            print(f"  [FOUND] {fn} ({mb} MB)")
        else:
            print(f"  [MISSING] {fn} (expected at {p})")

    # 7. ChatGPT Dynamic Config Recommendation
    if cpu_count <= 2:
        rec_threads = cpu_count
        thread_reason = f"System has {cpu_count} core(s). Use all available core(s)."
    elif cpu_count <= 4:
        rec_threads = 2
        thread_reason = f"System has {cpu_count} cores. Capping to 2 threads prevents thermal throttling on fanless/low-TDP chassis and leaves headroom for OS."
    elif cpu_count <= 8:
        rec_threads = min(4, cpu_count - 1)
        thread_reason = f"System has {cpu_count} cores. Using {rec_threads} threads for optimal PyTorch parallel efficiency without context thrashing."
    else:
        rec_threads = min(8, max(4, cpu_count // 2))
        thread_reason = f"System has {cpu_count} cores. Using {rec_threads} threads (higher thread counts on CPU GNNs exhibit diminishing returns and high memory per thread)."

    if ram_gb <= 4.5:
        ram_profile = "Low RAM (~4 GB)"
        rec_max_domains = 15000
        rec_max_per_class = 7500
        rec_batch_size = 128
        rec_hidden_dim = 16
        rec_epochs = 40
        rec_seeds = "[42, 1337]"
    elif ram_gb <= 8.5:
        ram_profile = "Standard Thin Client (~8 GB)"
        rec_max_domains = 30000
        rec_max_per_class = 15000
        rec_batch_size = 256
        rec_hidden_dim = 32
        rec_epochs = 50
        rec_seeds = "[42, 1337, 2024]"
    elif ram_gb <= 16.5:
        ram_profile = "High-Spec Thin Client / Workstation (~16 GB)"
        rec_max_domains = 50000
        rec_max_per_class = 25000
        rec_batch_size = 512
        rec_hidden_dim = 48
        rec_epochs = 60
        rec_seeds = "[42, 1337, 2024]"
    else:
        ram_profile = "Large Memory System (>= 32 GB)"
        rec_max_domains = 100000
        rec_max_per_class = 50000
        rec_batch_size = 1024
        rec_hidden_dim = 64
        rec_epochs = 100
        rec_seeds = "[42, 1337, 2024, 777, 999]"

    # Generate dynamic active config automatically
    try:
        from src.hdg.config import generate_dynamic_config
        active_yaml = generate_dynamic_config(verbose=False)
        print("\n" + "=" * 72)
        print("  AUTOMATIC DYNAMIC CONFIGURATION GENERATED")
        print("=" * 72)
        print(f"  Configuration file written to: {active_yaml}")
        print(f"  No manual editing is required! Simply run:")
        print(f"    bash pipeline/run_real.sh")
        print("=" * 72)
    except Exception as e:
        print(f"\n[Note] Could not auto-generate active config: {e}")





if __name__ == "__main__":
    main()
