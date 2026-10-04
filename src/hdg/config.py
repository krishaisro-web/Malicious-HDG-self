"""
Configuration loader, path validation, and run management for Malicious-HDG.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import yaml


@dataclass
class ResolvedPaths:
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    checkpoints_dir: Path
    is_fixture: bool
    is_smoke: bool = False

    def check_write_path(self, target_path: Path) -> None:
        """Enforces that fixture runs never write to real results."""
        resolved = str(target_path.resolve()).replace("\\", "/")
        if self.is_fixture:
            allowed_tokens = [
                "/results/fixture",
                "/results_fixture",
                "/data/fixture",
                "/data/processed/fixture",
                "/data_fixture",
                "/artifacts/checkpoints/fixture",
            ]
            if not any(token in resolved for token in allowed_tokens):
                raise PermissionError(
                    f"[FIXTURE SAFEGUARD] Refusing to write fixture output to production path: {resolved}. "
                    f"Fixture output must only go to results/fixture/, data/processed/fixture/, or artifacts/checkpoints/fixture/."
                )


def detect_system_hardware() -> Dict[str, Any]:
    """
    Dynamically probes system hardware specs (RAM, Swap, CPU cores, GPU)
    using psutil with fallbacks for /proc/meminfo and os.cpu_count().
    """
    import platform
    import sys

    cpu_cores = os.cpu_count() or 1
    total_ram_gb = 0.0
    avail_ram_gb = 0.0
    total_swap_gb = 0.0
    free_swap_gb = 0.0

    # 1. Probe via psutil if available
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        total_ram_gb = round(vm.total / (1024**3), 2)
        avail_ram_gb = round(vm.available / (1024**3), 2)
        total_swap_gb = round(sm.total / (1024**3), 2)
        free_swap_gb = round(sm.free / (1024**3), 2)
    except Exception:
        pass

    # 2. Fallback to /proc/meminfo on Linux
    if total_ram_gb == 0.0:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                mem_dict = {}
                with open(meminfo, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            mem_dict[parts[0].strip()] = parts[1].strip()
                tot_kb = int(mem_dict.get("MemTotal", "0 kB").split()[0])
                avail_kb = int(mem_dict.get("MemAvailable", mem_dict.get("MemFree", "0 kB")).split()[0])
                swap_kb = int(mem_dict.get("SwapTotal", "0 kB").split()[0])
                swap_free_kb = int(mem_dict.get("SwapFree", "0 kB").split()[0])
                total_ram_gb = round(tot_kb / (1024**2), 2)
                avail_ram_gb = round(avail_kb / (1024**2), 2)
                total_swap_gb = round(swap_kb / (1024**2), 2)
                free_swap_gb = round(swap_free_kb / (1024**2), 2)
            except Exception:
                pass

    # 3. Default fallback if RAM cannot be detected
    if total_ram_gb == 0.0:
        total_ram_gb = 8.0
        avail_ram_gb = 6.0

    # 4. GPU detection
    cuda_available = False
    gpu_name = None
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "cpu_cores": cpu_cores,
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "total_swap_gb": total_swap_gb,
        "free_swap_gb": free_swap_gb,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version.split()[0],
    }


def is_thin_client_environment() -> bool:
    """
    Detects if the environment matches an HP Thin Client or resource-constrained CPU system.
    If physical RAM is > 32 GB (e.g. ISRO 245 GB compute server), it is an enterprise server,
    NEVER a thin client.
    """
    if os.environ.get("HDG_PROFILE") in ("thin_client", "hp_thin_client"):
        return True

    hw = detect_system_hardware()
    if hw["total_ram_gb"] > 32.0:
        return False
    if hw["total_ram_gb"] <= 16.5:
        return True

    return hw["cpu_cores"] <= 4


def generate_dynamic_config(
    base_config_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    hw_override: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Path:
    """
    Dynamically probes system hardware and generates a tuned configuration YAML.
    Calibrates thread counts, batch sizes, graph domain limits, hidden dimensions,
    and seed budgets based on available RAM and CPU cores:
      - >= 64 GB RAM (e.g. 245 GB ISRO Server): Full dataset, batch size 1024, 200 epochs, 5 seeds, scaled threads
      - 16 - 64 GB RAM (Workstation): 60k domains, batch size 512, 100 epochs, 3 seeds
      - 6 - 16 GB RAM (Standard Thin Client): 30k domains, batch size 256, 50 epochs, 2 threads
      - <= 6 GB RAM (Constrained Thin Client): 15k domains, batch size 128, 40 epochs, 2 threads
    Saves to output_path (default: configs/active_config.yaml) and returns the Path.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if base_config_path is None:
        base_config_path = repo_root / "configs" / "default.yaml"
    else:
        base_config_path = Path(base_config_path)
        if not base_config_path.is_absolute():
            base_config_path = repo_root / base_config_path

    if output_path is None:
        output_path = repo_root / "configs" / "active_config.yaml"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = repo_root / output_path

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hw = hw_override or detect_system_hardware()
    ram_gb = hw["total_ram_gb"]
    avail_gb = hw["avail_ram_gb"]
    cpu_cores = hw["cpu_cores"]
    swap_gb = hw["total_swap_gb"]
    cuda_avail = hw.get("cuda_available", False)

    # Classify hardware tier
    if ram_gb >= 64.0:
        profile_name = "enterprise_server"
        profile_desc = f"Enterprise Multi-Core Server (~{round(ram_gb)} GB RAM Mode, Full Dataset)"
        num_threads = min(32, max(4, cpu_cores))
        max_domains = 100000
        max_per_class = None  # None becomes null in YAML, taking all records
        hub_degree = 500
        hub_percentile = 99.9
        batch_size = 1024
        epochs = 200
        patience = 20
        hidden_dim = 64
        out_dim = 32
        num_heads = 4
        seeds = [42, 1337, 2024, 777, 999]
        tfidf_features = 5000
        xgb_estimators = 100
        xgb_depth = 6
        bootstrap_samples = 500
        latency_queries = 500
        latency_warmup = 20
        memory_limit_gb = None
    elif ram_gb > 16.0:
        profile_name = "workstation"
        profile_desc = f"Mid-Range Workstation Profile ({round(ram_gb)} GB RAM Mode)"
        num_threads = min(16, max(4, cpu_cores))
        max_domains = 60000
        max_per_class = 30000
        hub_degree = 400
        hub_percentile = 99.7
        batch_size = 512
        epochs = 100
        patience = 15
        hidden_dim = 48
        out_dim = 24
        num_heads = 4
        seeds = [42, 1337, 2024]
        tfidf_features = 3500
        xgb_estimators = 75
        xgb_depth = 5
        bootstrap_samples = 300
        latency_queries = 300
        latency_warmup = 15
        memory_limit_gb = round(ram_gb * 0.75, 1)
    elif ram_gb > 6.0:
        profile_name = "hp_thin_client_standard"
        profile_desc = f"Standard HP Thin Client Profile ({round(ram_gb)} GB RAM Mode, 2 Threads Safe)"
        num_threads = 2
        max_domains = 30000
        max_per_class = 15000
        hub_degree = 300
        hub_percentile = 99.5
        batch_size = 256
        epochs = 50
        patience = 10
        hidden_dim = 32
        out_dim = 16
        num_heads = 2
        seeds = [42, 1337, 2024]
        tfidf_features = 2500
        xgb_estimators = 50
        xgb_depth = 4
        bootstrap_samples = 200
        latency_queries = 200
        latency_warmup = 10
        memory_limit_gb = 8.0
    else:
        profile_name = "hp_thin_client_constrained"
        profile_desc = f"Constrained HP Thin Client Profile (~{round(ram_gb)} GB RAM Mode, 2 Threads Safe)"
        num_threads = 2
        max_domains = 15000
        max_per_class = 7500
        hub_degree = 200
        hub_percentile = 99.0
        batch_size = 128
        epochs = 40
        patience = 8
        hidden_dim = 16
        out_dim = 8
        num_heads = 2
        seeds = [42, 1337]
        tfidf_features = 1500
        xgb_estimators = 30
        xgb_depth = 3
        bootstrap_samples = 100
        latency_queries = 100
        latency_warmup = 5
        memory_limit_gb = 4.0

    # Apply overrides to cfg
    cfg.setdefault("hardware", {})
    cfg["hardware"]["profile"] = profile_name
    cfg["hardware"]["profile_description"] = profile_desc
    cfg["hardware"]["num_threads"] = num_threads
    cfg["hardware"]["detected_ram_gb"] = ram_gb
    cfg["hardware"]["detected_cpu_cores"] = cpu_cores
    cfg["hardware"]["memory_limit_gb"] = memory_limit_gb

    cfg.setdefault("parsing", {})
    cfg["parsing"]["max_per_class"] = max_per_class

    cfg.setdefault("graph", {})
    cfg["graph"]["max_domains"] = max_domains
    cfg["graph"]["hub_degree"] = hub_degree
    cfg["graph"]["hub_percentile"] = hub_percentile

    cfg.setdefault("model", {})
    for mod_name in ["sage", "attn", "sage_guard"]:
        cfg["model"].setdefault(mod_name, {})
        cfg["model"][mod_name]["hidden_dim"] = hidden_dim
        cfg["model"][mod_name]["out_dim"] = out_dim
    cfg["model"]["attn"]["num_heads"] = num_heads

    cfg["model"].setdefault("training", {})
    cfg["model"]["training"]["epochs"] = epochs
    cfg["model"]["training"]["patience"] = patience
    cfg["model"]["training"]["batch_size"] = batch_size

    cfg.setdefault("randomness", {})
    cfg["randomness"]["evaluation_seeds"] = seeds

    cfg.setdefault("baselines", {})
    cfg["baselines"].setdefault("tfidf", {})["max_features"] = tfidf_features
    cfg["baselines"].setdefault("xgboost", {})["n_estimators"] = xgb_estimators
    cfg["baselines"]["xgboost"]["max_depth"] = xgb_depth

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["bootstrap_samples"] = bootstrap_samples

    cfg.setdefault("latency", {})
    cfg["latency"]["num_queries"] = latency_queries
    cfg["latency"]["warmup"] = latency_warmup

    header = (
        "# ==============================================================================\n"
        "# MALICIOUS-HDG DYNAMIC HARDWARE CONFIGURATION\n"
        "# Generated automatically by src/hdg/config.py\n"
        "#\n"
        f"# DETECTED SYSTEM SPECS:\n"
        f"#   OS Platform:       {hw.get('platform', 'Unknown')}\n"
        f"#   CPU Cores:         {cpu_cores} logical cores\n"
        f"#   Physical RAM:      {ram_gb:.1f} GB ({avail_gb:.1f} GB available)\n"
        f"#   Swap Space:        {swap_gb:.1f} GB\n"
        f"#   CUDA / GPU:        {'Available (' + str(hw.get('gpu_name')) + ')' if cuda_avail else 'CPU-Only (No GPU)'}\n"
        f"#\n"
        f"# DYNAMIC CALIBRATION:\n"
        f"#   Assigned Profile:  {profile_name}\n"
        f"#   Description:       {profile_desc}\n"
        f"#   Assigned Threads:  {num_threads}\n"
        f"#   Max Domains:       {max_domains}\n"
        f"#   Max Per Class:     {max_per_class if max_per_class is not None else 'null (Full Dataset)'}\n"
        f"#   Batch Size:        {batch_size}\n"
        f"#   Hidden Dimension:  {hidden_dim}\n"
        f"#   Epochs:            {epochs}\n"
        f"#   Evaluation Seeds:  {seeds}\n"
        "# ==============================================================================\n\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    if verbose:
        print("=" * 78)
        print("  MALICIOUS-HDG: DYNAMIC CONFIGURATION GENERATED")
        print("=" * 78)
        print(f"  Detected RAM:      {ram_gb:.1f} GB ({avail_gb:.1f} GB available)")
        print(f"  Detected CPU:      {cpu_cores} logical cores")
        print(f"  Detected Swap:     {swap_gb:.1f} GB")
        print(f"  Assigned Profile:  {profile_name}")
        print(f"  Profile Details:   {profile_desc}")
        print(f"  Active Config:     {output_path.resolve()}")
        print(f"  Allocated Threads: {num_threads} (OMP/MKL/OpenBLAS synchronized)")
        print(f"  Graph Max Domains: {max_domains}")
        print(f"  Parsing Cap:       {max_per_class or 'Full Dataset (Unrestricted)'}")
        print(f"  Batch Size:        {batch_size} | Epochs: {epochs} | Seeds: {seeds}")
        print("=" * 78)

    return output_path


def init_thread_pool(cfg: Optional[Dict[str, Any]] = None) -> int:
    """
    Sets PyTorch thread count from HDG_THREADS, config, or cpu_count().
    Ensures OpenMP / MKL / BLAS libraries do not oversubscribe threads,
    which is critical on low-power HP Thin Clients (2-4 cores) to prevent
    thermal throttling, severe context switching, and high per-thread memory.
    Returns configured thread count.
    """
    env_threads = os.environ.get("HDG_THREADS")
    if env_threads is not None:
        num_threads = int(env_threads)
    elif cfg is not None and "threads" in cfg and cfg["threads"] is not None:
        num_threads = int(cfg["threads"])
    elif cfg is not None and "hardware" in cfg and "num_threads" in cfg["hardware"] and cfg["hardware"]["num_threads"] is not None:
        num_threads = int(cfg["hardware"]["num_threads"])
    else:
        avail = os.cpu_count() or 1
        # On thin client or low-core machines (<= 4 cores and <= 16GB RAM), cap to min(2, avail)
        # to ensure thermal stability on passive cooling and leave headroom for OS.
        if is_thin_client_environment():
            num_threads = min(2, avail)
        else:
            # On large multi-core server (e.g. 245 GB RAM server with 16-128 cores),
            # utilize up to 32 threads
            num_threads = min(32, max(4, avail))

    torch.set_num_threads(num_threads)

    # Export thread caps to linear algebra and OpenMP runtimes
    for env_var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        if env_var not in os.environ:
            os.environ[env_var] = str(num_threads)

    return num_threads


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration file.
    Precedence:
      1. Explicit config_path passed as parameter
      2. HDG_CONFIG environment variable
      3. HDG_PROFILE=thin_client (loads configs/hp_thin_client.yaml)
      4. configs/active_config.yaml (if generated)
      5. configs/default.yaml
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if config_path is None:
        env_config = os.environ.get("HDG_CONFIG")
        if env_config:
            config_path = env_config
        elif os.environ.get("HDG_PROFILE") in ("thin_client", "hp_thin_client"):
            config_path = str(repo_root / "configs" / "hp_thin_client.yaml")
        else:
            active_path = repo_root / "configs" / "active_config.yaml"
            if active_path.exists():
                config_path = str(active_path)
            else:
                default_path = repo_root / "configs" / "default.yaml"
                config_path = str(default_path)
    else:
        c_path = Path(config_path)
        if not c_path.is_absolute() and not c_path.exists():
            candidate = repo_root / config_path
            if candidate.exists():
                config_path = str(candidate)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg



def get_resolved_paths(
    cfg: Dict[str, Any],
    is_fixture: bool = False,
    is_smoke: bool = False,
    root_dir: Optional[Path] = None
) -> ResolvedPaths:
    """
    Returns resolved filesystem paths depending on whether fixture and/or smoke mode is enabled.
    Guarantees strict isolation between test/fixture/smoke environments and production data.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    p = cfg["paths"]
    if is_fixture:
        raw_dir = root_dir / p.get("fixture_raw_dir", "data/fixture/zenodo")
        processed_dir = root_dir / p.get("fixture_processed_dir", "data/processed/fixture")
        base_results = root_dir / p.get("fixture_results_dir", "results/fixture")
        checkpoints_dir = root_dir / p.get("fixture_checkpoints_dir", "artifacts/checkpoints/fixture")
        results_dir = base_results
    else:
        raw_dir = root_dir / p.get("raw_data_dir", "data/raw/zenodo")
        processed_dir = root_dir / p.get("processed_data_dir", "data/processed/real")
        base_results = root_dir / p.get("results_dir", "results/real")
        checkpoints_dir = root_dir / p.get("checkpoints_dir", "artifacts/checkpoints/real")
        results_dir = (base_results / "smoke") if is_smoke else base_results

    return ResolvedPaths(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        is_fixture=is_fixture,
        is_smoke=is_smoke
    )


def _resolve_results_dir(target: Union[Path, ResolvedPaths]) -> Path:
    if isinstance(target, ResolvedPaths):
        return target.results_dir
    return target


def get_run_output_path(results_dir: Union[Path, ResolvedPaths], run_name: str) -> Path:
    """Returns path for an individual experiment run JSON."""
    r_dir = _resolve_results_dir(results_dir)
    runs_dir = r_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir / f"{run_name}.json"


def should_skip_run(results_dir: Union[Path, ResolvedPaths], run_name: str, force: bool = False) -> bool:
    """Checks if run JSON already exists to allow resumability unless --force is specified."""
    if force:
        return False
    path = get_run_output_path(results_dir, run_name)
    return path.exists() and path.stat().st_size > 0


def save_run_result(results_dir: Union[Path, ResolvedPaths], run_name: str, data: Dict[str, Any]) -> Path:
    """Persists an individual run JSON for resumability and report aggregation."""
    path = get_run_output_path(results_dir, run_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Malicious-HDG Dynamic Hardware Configuration CLI")
    parser.add_argument("--generate-dynamic", action="store_true", help="Probe hardware and generate dynamic configuration YAML")
    parser.add_argument("--base-config", type=str, default=None, help="Base config YAML to inherit from (default: configs/default.yaml)")
    parser.add_argument("--output", type=str, default="configs/active_config.yaml", help="Output path for active config")
    cli_args = parser.parse_args()

    out = generate_dynamic_config(
        base_config_path=cli_args.base_config,
        output_path=cli_args.output,
        verbose=True
    )
    print(f"[DYNAMIC CONFIG] Generated successfully at: {out}")

