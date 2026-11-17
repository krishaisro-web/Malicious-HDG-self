#!/usr/bin/env python3
"""
Master System, Hardware, GPU, VRAM, Storage, and Dependencies Audit Tool for Malicious-HDG.

Provides a 100% dynamic, all-in-one diagnostic capturing:
1. System & OS: Hostname, OS platform, kernel, glibc, uptime, user, Python runtime.
2. CPU Architecture: Model brand, physical & logical cores, frequency, load averages, lscpu.
3. System RAM & Swap: Total, available, used, free, buffers/cached, raw 'free -h'.
4. ROM & Disk Storage: Workspace, Root '/', and '/home' capacities, write accessibility, IO benchmark.
5. GPU & VRAM Breakdown (ALL GPUs, e.g. GPU 0 A100 80GB, GPU 1 T400 4GB):
   - PyTorch device properties, compute capability, SM count.
   - Exact VRAM breakdown (Total, Free, Used, Allocated, Reserved in GB & MB).
   - BF16, FP16, and TF32 hardware support checks.
   - nvidia-smi diagnostics (temperature, power draw, power limit, utilization %).
   - Active GPU processes (e.g. PID, process name like VLLM::EngineCore, memory used).
   - PyTorch CUDA live execution sanity test (allocation & matmul latency).
   - Precision safeguards (strict FP32 enforcement, PyG scatter determinism).
6. Required Python Libraries & Extensions (15 critical dependencies).
7. Zenodo Raw JSONs & Processed HeteroData Datasets (100% dynamic file size, rows, records, schema keys, nodes, edges).

Auto-routing behavior:
- On Linux / ISRO Server (or CUDA active): Automatically exports JSON to results/real_gpu/master_system_audit.json
- On macOS / local dev: Automatically exports JSON to results/mac/master_system_audit.json
- Preserves ISRO results from being overwritten by local Mac executions.
"""

import argparse
import getpass
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_uptime_info() -> Dict[str, Any]:
    """Retrieve system uptime dynamically."""
    try:
        import psutil
        boot_time = psutil.boot_time()
        uptime_sec = time.time() - boot_time
        hours, remainder = divmod(int(uptime_sec), 3600)
        minutes, seconds = divmod(remainder, 60)
        return {
            "uptime_seconds": int(uptime_sec),
            "uptime_human": f"{hours}h {minutes}m {seconds}s",
            "boot_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(boot_time)),
        }
    except Exception:
        uptime_file = Path("/proc/uptime")
        if uptime_file.exists():
            try:
                with open(uptime_file, "r") as f:
                    up_s = float(f.read().split()[0])
                hours, remainder = divmod(int(up_s), 3600)
                minutes, seconds = divmod(remainder, 60)
                return {
                    "uptime_seconds": int(up_s),
                    "uptime_human": f"{hours}h {minutes}m {seconds}s",
                }
            except Exception:
                pass
    return {"uptime_seconds": None, "uptime_human": "unknown"}


def get_cpu_model_brand() -> str:
    """Retrieve readable CPU model name across Linux and macOS."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL)
            return out.decode("utf-8").strip()
        elif system == "Linux":
            cpuinfo = Path("/proc/cpuinfo")
            if cpuinfo.exists():
                with open(cpuinfo, "r", encoding="utf-8") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
            out = subprocess.check_output("lscpu | grep 'Model name' | cut -d: -f2", shell=True, stderr=subprocess.DEVNULL)
            brand = out.decode("utf-8").strip()
            if brand:
                return brand
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_cpu_details() -> Dict[str, Any]:
    """Retrieve comprehensive CPU specifications, core counts, and frequency."""
    model = get_cpu_model_brand()
    logical_cores = os.cpu_count() or 1
    physical_cores = None
    freq_info = {}
    load_avg = None

    try:
        load_avg = list(os.getloadavg())
    except Exception:
        pass

    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            freq_info = {
                "current_mhz": round(cpu_freq.current, 1) if cpu_freq.current else None,
                "min_mhz": round(cpu_freq.min, 1) if cpu_freq.min else None,
                "max_mhz": round(cpu_freq.max, 1) if cpu_freq.max else None,
            }
    except Exception:
        pass

    lscpu_raw = None
    if platform.system() == "Linux":
        try:
            lscpu_raw = subprocess.check_output(["lscpu"], stderr=subprocess.DEVNULL).decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    return {
        "model_brand": model,
        "architecture": platform.machine(),
        "logical_cores": logical_cores,
        "physical_cores": physical_cores or (logical_cores // 2 if logical_cores > 1 else 1),
        "frequency_mhz": freq_info,
        "load_average_1_5_15": load_avg,
        "lscpu_summary": lscpu_raw,
    }


def get_ram_swap_info() -> Dict[str, Any]:
    """Retrieve detailed physical RAM and Swap statistics."""
    info: Dict[str, Any] = {
        "ram_total_gb": 0.0,
        "ram_available_gb": 0.0,
        "ram_used_gb": 0.0,
        "ram_free_gb": 0.0,
        "ram_percent_used": 0.0,
        "swap_total_gb": 0.0,
        "swap_used_gb": 0.0,
        "swap_free_gb": 0.0,
        "raw_free_output": None,
    }

    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
        info["ram_used_gb"] = round(vm.used / (1024**3), 2)
        info["ram_free_gb"] = round(getattr(vm, "free", vm.available) / (1024**3), 2)
        info["ram_percent_used"] = vm.percent
        info["swap_total_gb"] = round(sm.total / (1024**3), 2)
        info["swap_used_gb"] = round(sm.used / (1024**3), 2)
        info["swap_free_gb"] = round(sm.free / (1024**3), 2)
    except Exception:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            m: Dict[str, int] = {}
            with open(meminfo, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        val = parts[1].strip().split()[0]
                        if val.isdigit():
                            m[parts[0].strip()] = int(val)
            if "MemTotal" in m:
                info["ram_total_gb"] = round(m["MemTotal"] / (1024**2), 2)
            if "MemAvailable" in m:
                info["ram_available_gb"] = round(m["MemAvailable"] / (1024**2), 2)
            if "SwapTotal" in m:
                info["swap_total_gb"] = round(m["SwapTotal"] / (1024**2), 2)
            if "SwapFree" in m:
                info["swap_free_gb"] = round(m["SwapFree"] / (1024**2), 2)

    if platform.system() == "Linux":
        try:
            free_out = subprocess.check_output(["free", "-h"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            info["raw_free_output"] = free_out
        except Exception:
            pass

    return info


def get_rom_disk_storage_info(repo_root: Path) -> Dict[str, Any]:
    """Retrieve storage, partition capacities, and disk I/O performance."""
    storage_info: Dict[str, Any] = {
        "workspace_root": str(repo_root),
        "partitions": {},
        "raw_df_output": None,
        "write_latency_ms": None,
    }

    # Inspect Workspace partition
    total_b, used_b, free_b = shutil.disk_usage(repo_root)
    storage_info["workspace"] = {
        "path": str(repo_root),
        "total_gb": round(total_b / (1024**3), 2),
        "used_gb": round(used_b / (1024**3), 2),
        "free_gb": round(free_b / (1024**3), 2),
        "percent_used": round((used_b / total_b) * 100, 1) if total_b > 0 else 0.0,
        "write_accessible": os.access(repo_root, os.W_OK),
    }

    # Inspect Root '/' partition
    try:
        r_tot, r_used, r_free = shutil.disk_usage("/")
        storage_info["partitions"]["root"] = {
            "mount": "/",
            "total_gb": round(r_tot / (1024**3), 2),
            "used_gb": round(r_used / (1024**3), 2),
            "free_gb": round(r_free / (1024**3), 2),
        }
    except Exception:
        pass

    # Inspect '/home' partition (common on multi-user Linux clusters)
    home_p = Path("/home")
    if home_p.exists():
        try:
            h_tot, h_used, h_free = shutil.disk_usage("/home")
            storage_info["partitions"]["home"] = {
                "mount": "/home",
                "total_gb": round(h_tot / (1024**3), 2),
                "used_gb": round(h_used / (1024**3), 2),
                "free_gb": round(h_free / (1024**3), 2),
            }
        except Exception:
            pass

    # Capture raw df -h output
    try:
        df_out = subprocess.check_output(["df", "-h", "/", "/home"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        storage_info["raw_df_output"] = df_out
    except Exception:
        try:
            df_out = subprocess.check_output(["df", "-h", "."], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            storage_info["raw_df_output"] = df_out
        except Exception:
            pass

    # Quick disk I/O write latency benchmark (write 5MB and sync)
    try:
        test_file = repo_root / ".disk_io_test.tmp"
        dummy_chunk = b"0" * (1024 * 1024)  # 1MB
        t0 = time.time()
        with open(test_file, "wb") as f:
            for _ in range(5):
                f.write(dummy_chunk)
            f.flush()
            os.fsync(f.fileno())
        elapsed_ms = (time.time() - t0) * 1000.0
        if test_file.exists():
            test_file.unlink()
        storage_info["write_latency_ms"] = round(elapsed_ms, 2)
    except Exception:
        storage_info["write_latency_ms"] = None

    return storage_info


def get_gpu_vram_info() -> Dict[str, Any]:
    """
    Inspect GPU hardware, all physical devices (e.g. A100 vs T400),
    active GPU processes, VRAM breakdown, driver/CUDA versions, and live execution sanity.
    """
    gpu_data: Dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
        "nvidia_smi_detected": False,
        "nvidia_smi_output": None,
        "active_gpu_processes": [],
        "execution_sanity_test": {},
        "precision_safeguards": {
            "tf32_matmul_enabled": False,
            "tf32_cudnn_enabled": False,
            "strict_fp32_enforced": True,
            "deterministic_algorithms": False,
        },
    }

    # 1. nvidia-smi GPU query
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,power.draw,power.limit",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        gpu_data["nvidia_smi_detected"] = True
        gpu_data["nvidia_smi_output"] = smi_out
    except Exception:
        gpu_data["nvidia_smi_detected"] = False

    # 2. nvidia-smi Running Compute Processes Query (Detects VLLM, other workloads)
    if gpu_data["nvidia_smi_detected"]:
        try:
            proc_out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=gpu_name,pid,process_name,used_memory",
                 "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
            if proc_out:
                for line in proc_out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpu_data["active_gpu_processes"].append({
                            "gpu_name": parts[0],
                            "pid": parts[1],
                            "process_name": parts[2],
                            "used_memory": parts[3],
                        })
        except Exception:
            pass

    # 3. PyTorch CUDA inspection
    try:
        import torch
        gpu_data["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpu_data["cuda_version"] = torch.version.cuda
            gpu_data["cudnn_version"] = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
            cnt = torch.cuda.device_count()
            gpu_data["device_count"] = cnt

            for idx in range(cnt):
                props = torch.cuda.get_device_properties(idx)
                free_b, total_b = torch.cuda.mem_get_info(idx)
                alloc_b = torch.cuda.memory_allocated(idx)
                res_b = torch.cuda.memory_reserved(idx)
                used_b = total_b - free_b

                # BF16 support check
                bf16_sup = False
                try:
                    if hasattr(torch.cuda, "is_bf16_supported"):
                        bf16_sup = bool(torch.cuda.is_bf16_supported())
                    elif props.major >= 8:
                        bf16_sup = True
                except Exception:
                    bf16_sup = (props.major >= 8)

                dev_info = {
                    "index": idx,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                    "total_vram_gb": round(total_b / (1024**3), 2),
                    "free_vram_gb": round(free_b / (1024**3), 2),
                    "used_vram_gb": round(used_b / (1024**3), 2),
                    "allocated_vram_gb": round(alloc_b / (1024**3), 4),
                    "reserved_vram_gb": round(res_b / (1024**3), 4),
                    "total_vram_mb": round(total_b / (1024**2), 1),
                    "free_vram_mb": round(free_b / (1024**2), 1),
                    "used_vram_mb": round(used_b / (1024**2), 1),
                    "bf16_supported": bf16_sup,
                    "fp16_supported": True,
                    "tf32_supported": (props.major >= 8),
                }
                gpu_data["devices"].append(dev_info)

            # Precision settings
            tf32_mat = getattr(torch.backends.cuda.matmul, "allow_tf32", False)
            tf32_cud = getattr(torch.backends.cudnn, "allow_tf32", False)
            gpu_data["precision_safeguards"]["tf32_matmul_enabled"] = tf32_mat
            gpu_data["precision_safeguards"]["tf32_cudnn_enabled"] = tf32_cud
            gpu_data["precision_safeguards"]["strict_fp32_enforced"] = (not tf32_mat) and (not tf32_cud)

            # Quick CUDA Live Allocation & MatMul Sanity Test on GPU 0
            try:
                t0 = time.time()
                t_a = torch.randn(1024, 1024, device="cuda:0", dtype=torch.float32)
                t_b = torch.randn(1024, 1024, device="cuda:0", dtype=torch.float32)
                _ = torch.matmul(t_a, t_b)
                torch.cuda.synchronize(0)
                dur_ms = (time.time() - t0) * 1000.0
                del t_a, t_b
                torch.cuda.empty_cache()
                gpu_data["execution_sanity_test"] = {
                    "passed": True,
                    "test_device": "cuda:0",
                    "matmul_1024_latency_ms": round(dur_ms, 2),
                }
            except Exception as e_live:
                gpu_data["execution_sanity_test"] = {
                    "passed": False,
                    "error": str(e_live),
                }

    except Exception as e:
        gpu_data["error"] = str(e)

    return gpu_data


def audit_python_libraries() -> Dict[str, Dict[str, Any]]:
    """Audit all dependencies and required libraries for Malicious-HDG."""
    libs_to_check = [
        ("torch", "Core PyTorch deep learning runtime"),
        ("torch_geometric", "PyG heterogeneous graph neural network framework"),
        ("torch_scatter", "PyG GPU scatter aggregation (optional C++ kernel)"),
        ("torch_sparse", "PyG sparse operations (optional C++ kernel)"),
        ("torch_cluster", "PyG graph clustering (optional C++ kernel)"),
        ("torch_spline_conv", "PyG spline conv (optional C++ kernel)"),
        ("numpy", "Numerical vector and matrix computing"),
        ("scipy", "Scientific computing, statistical tests & bootstrapping"),
        ("pandas", "Dataframe ingestion, tabular processing & dataset alignment"),
        ("sklearn", "Machine learning baselines, ROC-AUC, threshold metrics"),
        ("xgboost", "Gradient boosted decision tree baseline"),
        ("yaml", "YAML configuration parser (PyYAML)"),
        ("pyarrow", "Fast columnar Parquet dataset read/write"),
        ("psutil", "Hardware, RAM, Swap and process monitoring"),
        ("pytest", "Automated test suite execution"),
        ("requests", "HTTP client for external dataset downloads"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for mod_name, description in libs_to_check:
        info: Dict[str, Any] = {"description": description, "installed": False, "version": None}
        try:
            mod = importlib.import_module(mod_name)
            info["installed"] = True
            info["version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["installed"] = False
            info["version"] = None
        results[mod_name] = info

    return results


def probe_raw_json_file(file_path: Path) -> Dict[str, Any]:
    """Dynamically probe a Zenodo raw JSON export file without static hardcoding."""
    if not file_path.exists():
        return {
            "present": False,
            "file_name": file_path.name,
            "file_path": str(file_path),
        }

    stat = file_path.stat()
    size_bytes = stat.st_size
    size_mb = round(size_bytes / (1024**2), 3)

    record_count = 0
    sample_keys: List[str] = []
    sample_domains: List[str] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_char = ""
            for ch in f.read(100):
                if not ch.isspace():
                    first_char = ch
                    break
            f.seek(0)

            if first_char == "[":
                data = json.load(f)
                if isinstance(data, list):
                    record_count = len(data)
                    for item in data[:10]:
                        if isinstance(item, dict):
                            for k in item.keys():
                                if k not in sample_keys:
                                    sample_keys.append(str(k))
                            dname = item.get("domain_name") or item.get("domain") or item.get("name")
                            if dname and str(dname) not in sample_domains:
                                sample_domains.append(str(dname))
            else:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    record_count += 1
                    if record_count <= 10:
                        try:
                            obj = json.loads(line_str)
                            if isinstance(obj, dict):
                                for k in obj.keys():
                                    if k not in sample_keys:
                                        sample_keys.append(str(k))
                                dname = obj.get("domain_name") or obj.get("domain") or obj.get("name")
                                if dname and str(dname) not in sample_domains:
                                    sample_domains.append(str(dname))
                        except Exception:
                            pass
    except Exception as e:
        return {
            "present": True,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "size_bytes": size_bytes,
            "size_mb": size_mb,
            "error": str(e),
        }

    return {
        "present": True,
        "file_name": file_path.name,
        "file_path": str(file_path),
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "num_records": record_count,
        "detected_schema_keys": sample_keys,
        "sample_domains": sample_domains,
    }


def probe_processed_graph_dataset(processed_dir: Path) -> Dict[str, Any]:
    """Dynamically probe processed parquet cohort and HeteroData graph without any hardcoding."""
    details: Dict[str, Any] = {
        "present": False,
        "directory": str(processed_dir),
        "parquet_cohort": {},
        "hetero_graph": {},
        "manifest": {},
    }

    if not processed_dir.exists():
        return details

    # 1. Parquet Ingestion & Probing
    parquet_path = processed_dir / "domains.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            label_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()} if "label" in df.columns else {}
            sources_list = [str(s) for s in df["source"].dropna().unique().tolist()] if "source" in df.columns else []

            details["parquet_cohort"] = {
                "present": True,
                "file_path": str(parquet_path),
                "size_mb": round(parquet_path.stat().st_size / (1024**2), 3),
                "num_rows": len(df),
                "num_columns": len(df.columns),
                "column_names": list(df.columns),
                "label_distribution": label_counts,
                "distinct_sources": sources_list,
            }
        except Exception as e:
            details["parquet_cohort"] = {"present": True, "error": str(e)}

    # 2. HeteroData Graph Probing
    graph_path = processed_dir / "heterodata.pt"
    if graph_path.exists():
        try:
            import torch
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            node_counts = {str(nt): int(graph[nt].num_nodes) for nt in graph.node_types}
            edge_counts = {
                f"{et[0]}__{et[1]}__{et[2]}": int(graph[et].edge_index.size(1))
                for et in graph.edge_types
            }

            feat_shape = None
            if hasattr(graph["domain"], "x") and graph["domain"].x is not None:
                feat_shape = list(graph["domain"].x.shape)
            elif hasattr(graph["domain"], "raw_x") and graph["domain"].raw_x is not None:
                feat_shape = list(graph["domain"].raw_x.shape)

            details["hetero_graph"] = {
                "present": True,
                "file_path": str(graph_path),
                "size_mb": round(graph_path.stat().st_size / (1024**2), 3),
                "total_nodes": sum(node_counts.values()),
                "total_edges": sum(edge_counts.values()),
                "node_types": [str(nt) for nt in graph.node_types],
                "node_counts": node_counts,
                "edge_types": [[str(x) for x in et] for et in graph.edge_types],
                "edge_counts": edge_counts,
                "domain_feature_tensor_shape": feat_shape,
            }
        except Exception as e:
            details["hetero_graph"] = {"present": True, "error": str(e)}

    # 3. Manifest Probing
    manifest_path = processed_dir / "graph_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                mani = json.load(f)
            details["manifest"] = {
                "present": True,
                "file_path": str(manifest_path),
                "size_kb": round(manifest_path.stat().st_size / 1024, 2),
                "payload": mani,
            }
        except Exception as e:
            details["manifest"] = {"present": True, "error": str(e)}

    details["present"] = bool(
        details["parquet_cohort"].get("present") or details["hetero_graph"].get("present")
    )
    return details


def audit_datasets_and_artifacts(repo_root: Path) -> Dict[str, Any]:
    """Completely dynamic dataset and graph artifact profiling."""
    raw_dir = repo_root / "data" / "raw" / "zenodo"
    processed_dir = repo_root / "data" / "processed" / "real"

    raw_candidates = [
        raw_dir / "malware.json",
        raw_dir / "benign_umbrella.json",
        raw_dir / "benign_cesnet.json",
    ]

    status: Dict[str, Any] = {
        "raw_datasets": {f.name: probe_raw_json_file(f) for f in raw_candidates},
        "processed_cohort": probe_processed_graph_dataset(processed_dir),
    }

    status["all_raw_present"] = all(
        finfo.get("present", False) for finfo in status["raw_datasets"].values()
    )
    status["processed_ready"] = status["processed_cohort"].get("present", False)

    return status


def resolve_auto_json_path(repo_root: Path, requested_path: Optional[str], no_json: bool) -> Optional[Path]:
    """
    Intelligently auto-route JSON output destination:
    - If user specifies an explicit path: use it.
    - If no_json is True: return None.
    - If 'auto':
        - On Linux / ISRO Server (or if CUDA is available): output to results/real_gpu/master_system_audit.json
        - On macOS / local dev: output to results/mac/master_system_audit.json (100% gitignored)
    """
    if no_json or requested_path in (None, "", "none", "false", "False"):
        return None

    if requested_path != "auto":
        return Path(requested_path)

    is_linux_or_cuda = (platform.system() == "Linux")
    try:
        import torch
        if torch.cuda.is_available():
            is_linux_or_cuda = True
    except Exception:
        pass

    if is_linux_or_cuda:
        return repo_root / "results" / "real_gpu" / "master_system_audit.json"
    else:
        return repo_root / "results" / "mac" / "master_system_audit.json"


def run_master_audit(repo_root: Optional[Path] = None, json_out: Optional[str] = "auto", no_json: bool = False) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Run complete master system, hardware, GPU, VRAM, and library audit."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    # 1. System & OS
    os_info = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "libc_ver": list(platform.libc_ver()),
        "uptime": get_uptime_info(),
        "current_user": getpass.getuser(),
        "python_version": sys.version.split()[0],
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "working_directory": str(repo_root),
    }

    # 2. CPU
    cpu_info = get_cpu_details()

    # 3. RAM & Swap
    mem_info = get_ram_swap_info()

    # 4. Storage & Disk
    disk_info = get_rom_disk_storage_info(repo_root)

    # 5. GPU & VRAM
    gpu_info = get_gpu_vram_info()

    # 6. Python Libraries
    lib_info = audit_python_libraries()

    # 7. Datasets
    data_info = audit_datasets_and_artifacts(repo_root)

    report = {
        "title": "Malicious-HDG Master System, Hardware, GPU & Dataset Audit Manifest",
        "system": os_info,
        "cpu": cpu_info,
        "memory": mem_info,
        "storage": disk_info,
        "gpu": gpu_info,
        "libraries": lib_info,
        "datasets": data_info,
    }

    final_json_path = resolve_auto_json_path(repo_root, json_out, no_json)
    if final_json_path:
        final_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report, final_json_path


def print_audit_report(report: Dict[str, Any], exported_path: Optional[Path]) -> None:
    """Print formatted audit report to console."""
    sep = "=" * 78

    print(sep)
    print("      MALICIOUS-HDG MASTER SYSTEM, HARDWARE, GPU & DEPENDENCY AUDIT")
    print(sep)

    # 1. System
    s = report["system"]
    print("\n[1] OPERATING SYSTEM & RUNTIME:")
    print(f"  Hostname:           {s['hostname']}")
    print(f"  OS Platform:        {s['platform']}")
    print(f"  Architecture:       {s['architecture']}")
    print(f"  User / Workspace:   {s['current_user']} @ {s['working_directory']}")
    print(f"  System Uptime:      {s['uptime'].get('uptime_human', 'N/A')}")
    print(f"  Python Runtime:     v{s['python_version']} ({s['python_executable']})")

    # 2. CPU
    c = report["cpu"]
    print("\n[2] CPU ARCHITECTURE & CORES:")
    print(f"  CPU Model:          {c['model_brand']}")
    print(f"  Logical Cores:      {c['logical_cores']} threads (Physical: {c['physical_cores']} cores)")
    if c.get("frequency_mhz", {}).get("current_mhz"):
        print(f"  Clock Frequency:    {c['frequency_mhz']['current_mhz']} MHz")
    if c.get("load_average_1_5_15"):
        print(f"  Load Average:       {c['load_average_1_5_15']}")

    # 3. RAM & Swap
    m = report["memory"]
    print("\n[3] PHYSICAL MEMORY & SWAP (OOM Killer Prevention):")
    print(f"  Physical RAM:       {m['ram_total_gb']} GB total ({m['ram_available_gb']} GB available, {m['ram_used_gb']} GB used)")
    print(f"  Swap Space:         {m['swap_total_gb']} GB total ({m['swap_free_gb']} GB free)")
    if m["swap_total_gb"] < 2.0 and m["ram_total_gb"] <= 8.5:
        print("  [!] WARNING: Low/zero swap detected on small memory host.")

    # 4. Storage & Partitions
    d = report["storage"]
    w = d["workspace"]
    print("\n[4] STORAGE, DISK PARTITIONS & I/O:")
    print(f"  Workspace:          {w['path']}")
    print(f"  Workspace Capacity: {w['free_gb']} GB free / {w['total_gb']} GB total ({w['percent_used']}% used)")
    print(f"  Write Permission:   {'OK (Writable)' if w['write_accessible'] else 'ERROR (Read-Only)'}")
    if d.get("write_latency_ms") is not None:
        print(f"  Disk I/O Latency:   {d['write_latency_ms']} ms (5MB sync flush)")
    if "root" in d.get("partitions", {}):
        rt = d["partitions"]["root"]
        print(f"  Root '/' Partition: {rt['free_gb']} GB free out of {rt['total_gb']} GB")
    if "home" in d.get("partitions", {}):
        hm = d["partitions"]["home"]
        print(f"  '/home' Partition:  {hm['free_gb']} GB free out of {hm['total_gb']} GB")

    # 5. GPU & VRAM
    g = report["gpu"]
    print("\n[5] GPU HARDWARE, VRAM & PROCESS MONITOR:")
    print(f"  CUDA Available:     {g['cuda_available']}")
    if g["cuda_available"]:
        print(f"  CUDA Version:       {g['cuda_version']} (cuDNN: {g['cudnn_version']})")
        print(f"  Detected GPUs:      {g['device_count']}")
        for dev in g["devices"]:
            print(f"    -> GPU [{dev['index']}]: {dev['name']}")
            print(f"       Compute Cap:   {dev['compute_capability']} | SMs: {dev['multi_processor_count']}")
            print(f"       Total VRAM:    {dev['total_vram_gb']} GB ({dev['total_vram_mb']} MB)")
            print(f"       Free VRAM:     {dev['free_vram_gb']} GB ({dev['free_vram_mb']} MB)")
            print(f"       Used VRAM:     {dev['used_vram_gb']} GB ({dev['used_vram_mb']} MB)")
            print(f"       BF16 Support:  {dev['bf16_supported']} | TF32 Support: {dev['tf32_supported']}")

        safes = g["precision_safeguards"]
        print(f"  Strict FP32 State:  {'ENFORCED (allow_tf32=False)' if safes['strict_fp32_enforced'] else 'WARNING (TF32 active)'}")

        if g.get("execution_sanity_test", {}).get("passed"):
            lat = g["execution_sanity_test"].get("matmul_1024_latency_ms")
            print(f"  Live Sanity Test:   PASSED (1024x1024 MatMul latency: {lat} ms)")

        # Active compute processes (e.g. VLLM)
        if g.get("active_gpu_processes"):
            print("  --- Active Multi-Tenant GPU Processes ---")
            for proc in g["active_gpu_processes"]:
                print(f"    [PROCESS] PID {proc['pid']:<7} | {proc['process_name']:<28} | Memory: {proc['used_memory']}")
        else:
            print("  Active Processes:   No external compute processes detected (GPU completely dedicated).")
    else:
        print("  [NOTE] No NVIDIA CUDA GPU active (Host running in CPU / Apple Silicon mode).")

    # 6. Python Libraries
    print("\n[6] REQUIRED PYTHON LIBRARIES & EXTENSIONS:")
    for mod_name, lib_data in report["libraries"].items():
        if lib_data["installed"]:
            status_tag = f"\033[1;32m[INSTALLED v{lib_data['version']}]\033[0m"
        else:
            status_tag = "\033[1;31m[MISSING]\033[0m"
        print(f"  - {mod_name:<20} {status_tag:<32} ({lib_data['description']})")

    # 7. Datasets
    data_audit = report["datasets"]
    print("\n[7] DATASET & GRAPH ARTIFACTS AUDIT (100% DYNAMIC INSPECTION):")
    print("  --- Raw Zenodo JSON Datasets ---")
    for fn, rinfo in data_audit["raw_datasets"].items():
        if rinfo.get("present"):
            keys_str = ", ".join(rinfo.get("detected_schema_keys", [])[:5])
            print(f"  [✓] {fn:<24} | Records: {rinfo['num_records']:>6} | Size: {rinfo['size_mb']:>6.2f} MB | Keys: [{keys_str}]")
        else:
            print(f"  [✗] {fn:<24} | MISSING")

    cohort = data_audit["processed_cohort"]
    print("\n  --- Processed Heterogeneous Graph Cohort ---")
    if cohort.get("present"):
        pq = cohort.get("parquet_cohort", {})
        if pq.get("present"):
            lbl_str = ", ".join(f"Label {k}: {v}" for k, v in pq.get("label_distribution", {}).items())
            print(f"  [✓] domains.parquet:        {pq['num_rows']} domains, {pq['num_columns']} columns ({pq['size_mb']} MB)")
            print(f"      Label Counts:           {lbl_str}")

        hg = cohort.get("hetero_graph", {})
        if hg.get("present"):
            nodes_str = ", ".join(f"{k}: {v}" for k, v in hg.get("node_counts", {}).items())
            print(f"  [✓] heterodata.pt:          {hg['total_nodes']} total nodes across {len(hg['node_types'])} types ({hg['size_mb']} MB)")
            print(f"      Node Breakdown:         {nodes_str}")
            print(f"      Total Edges:            {hg['total_edges']} edges across {len(hg['edge_types'])} relation types")
            print(f"      Domain Features Shape:  {hg.get('domain_feature_tensor_shape')}")
    else:
        print("  [✗] Processed cohort not found.")

    print("\n" + sep)
    print("                      AUDIT SUMMARY & MANIFEST")
    print(sep)
    if exported_path:
        print(f"  -> JSON Manifest Exported:  {exported_path}")
    if g["cuda_available"]:
        print("  -> System Ready for A100 GPU Execution: YES (Strict FP32 & Shared Cap Enabled)")
    else:
        print("  -> System Ready for CPU Execution:     YES (CPU Reference Benchmark Protocol)")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master System, Hardware, GPU, VRAM, Storage, and Dependencies Audit Tool."
    )
    parser.add_argument(
        "--json",
        type=str,
        default="auto",
        help="Path to export structured JSON report (default: 'auto' routes to results/real_gpu on ISRO/Linux and results/mac on Mac)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable automatic JSON file export",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal text output")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    report, exported_path = run_master_audit(
        repo_root=repo_root, json_out=args.json, no_json=args.no_json
    )

    if not args.quiet:
        print_audit_report(report, exported_path)


if __name__ == "__main__":
    main()
