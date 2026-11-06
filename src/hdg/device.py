"""
Centralized device resolution, precision enforcement, and hardware provenance.

Adheres strictly to the nine mandatory GPU constraints in docs/GPU_ADAPTATION_PLAN.md:
1. Strict CPU default (never auto-switch).
2. Strict FP32 enforcement (disable TF32 / BF16).
3. PyG scatter-add determinism mode.
4. Shared-GPU memory fraction capping and dynamic pre-flight VRAM verification.
5. Execution provenance logging across all runs.
"""

import os
import platform
import socket
import subprocess
from typing import Any, Dict, Optional, Union

import torch


def resolve_device(
    device_arg: Optional[str] = None,
    device_str: Optional[str] = None,
) -> torch.device:
    """
    Resolve device string to torch.device with strict defaults and FP32 enforcement.

    Parameters
    ----------
    device_arg : Optional[str]
        Explicit device string ('cpu', 'cuda', 'cuda:0', 'mps').
    device_str : Optional[str]
        Backwards-compatible alias for device_arg.
        If both None, checks HDG_DEVICE environment variable.
        If still None, defaults strictly to 'cpu' (Constraint #2).

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """
    target = device_arg if device_arg is not None else device_str
    if target is None or target.strip() == "":
        target = os.environ.get("HDG_DEVICE", "cpu")

    dev_clean = target.strip().lower()

    if dev_clean == "cpu":
        return torch.device("cpu")

    if dev_clean.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested ('{target}') but torch.cuda.is_available() is False. "
                "GPU execution requires a valid NVIDIA driver and CUDA toolkit."
            )

        device = torch.device(dev_clean)

        # Enforce Constraint #3: Strict FP32 precision (disable TF32 / BF16)
        # Prevents logit ties at critical threshold ~2.6, protects tail metrics at FPR <= 0.1%,
        # and preserves GNNGuard cosine pruning resolution (eps = 1e-8).
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

        # Enforce Constraint #5: Deterministic algorithms for CUDA scatter operations
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

        return device

    if dev_clean == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS device requested but torch.backends.mps.is_available() is False.")

    raise ValueError(
        f"Unsupported device '{target}'. Valid options are 'cpu', 'cuda', 'cuda:N', or 'mps'."
    )


def set_gpu_memory_limits(
    fraction: float = 0.35,
    device: Optional[Union[torch.device, int, str]] = None,
) -> None:
    """
    Cap per-process GPU memory fraction adhering to Constraint #8.

    If CUDA is active, calls torch.cuda.set_per_process_memory_fraction(fraction, device=device).
    """
    if not torch.cuda.is_available():
        return

    dev_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, int):
        dev_idx = device
    elif isinstance(device, str):
        dev = torch.device(device)
        if dev.type != "cuda":
            return
        dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
    else:
        dev_idx = torch.cuda.current_device()

    torch.cuda.set_per_process_memory_fraction(fraction, dev_idx)


def check_vram_available(
    min_free_gb: float = 8.0,
    device: Optional[Union[torch.device, int, str]] = None,
) -> bool:
    """
    Dynamically inspects torch.cuda.mem_get_info() to ensure free VRAM >= min_free_gb.

    Returns True if free VRAM meets or exceeds min_free_gb, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    dev_idx = 0
    if isinstance(device, torch.device):
        if device.type != "cuda":
            return False
        dev_idx = device.index if device.index is not None else 0
    elif isinstance(device, int):
        dev_idx = device
    elif isinstance(device, str):
        dev = torch.device(device)
        if dev.type != "cuda":
            return False
        dev_idx = dev.index if dev.index is not None else 0

    try:
        free_bytes, _ = torch.cuda.mem_get_info(dev_idx)
        free_gb = free_bytes / (1024 ** 3)
        return free_gb >= min_free_gb
    except Exception:
        return False


def setup_shared_gpu_environment(
    dev_idx: int = 0,
    memory_fraction: float = 0.35,
    min_free_vram_gb: float = 8.0,
) -> Dict[str, Any]:
    """
    Configure shared GPU environment safeguards adhering to Constraint #8.

    Caps memory fraction to 35% (~28 GB on A100-80GB) to prevent out-of-memory collisions
    with other multi-tenant processes, and verifies dynamic free VRAM prior to execution.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("setup_shared_gpu_environment called but CUDA is not available.")

    # 1. Enforce strict FP32 precision
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    # 2. Set deterministic scatter mode
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # 3. Dynamic pre-seed VRAM verification
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    if free_gb < min_free_vram_gb:
        raise RuntimeError(
            f"Insufficient VRAM on CUDA device {dev_idx}: {free_gb:.2f} GB free out of {total_gb:.2f} GB. "
            f"Minimum required: {min_free_vram_gb:.2f} GB. Aborting to avoid OOM collision."
        )

    # 4. Cap per-process memory allocation
    set_gpu_memory_limits(memory_fraction, dev_idx)

    return {
        "device_index": dev_idx,
        "device_name": torch.cuda.get_device_name(dev_idx),
        "memory_fraction_cap": memory_fraction,
        "free_vram_gb": round(free_gb, 3),
        "total_vram_gb": round(total_gb, 3),
        "tf32_disabled": True,
        "deterministic_algorithms": True,
    }


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash, or 'unknown' if not in git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def get_provenance_metadata(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Record execution provenance metadata adhering to Constraint #2.

    Emitted into all run output JSON files to guarantee auditable lineage between
    CPU reference runs and GPU execution runs.
    """
    dev_str = str(device) if device is not None else "cpu"
    dev_type = device.type if device is not None else "cpu"

    provenance = {
        "device": dev_str,
        "device_type": dev_type,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "git_commit": get_git_commit_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "precision": "FP32",
        "tf32_matmul_enabled": getattr(torch.backends.cuda.matmul, "allow_tf32", False),
        "tf32_cudnn_enabled": getattr(torch.backends.cudnn, "allow_tf32", False),
    }

    if dev_type == "cuda" and torch.cuda.is_available():
        dev_idx = device.index if device.index is not None else 0
        provenance["device_name"] = torch.cuda.get_device_name(dev_idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        provenance["vram_free_gb"] = round(free_bytes / (1024 ** 3), 3)
        provenance["vram_total_gb"] = round(total_bytes / (1024 ** 3), 3)
    else:
        provenance["device_name"] = platform.processor() or "CPU"

    return provenance


# Backwards compatible aliases
get_execution_provenance = get_provenance_metadata
get_device_provenance = get_provenance_metadata


def set_cuda_precision_and_determinism() -> None:
    """Enforce strict FP32 precision and PyG CUDA scatter determinism."""
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Return memory statistics for CUDA device in gigabytes, or zeros if CPU."""
    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
        }

    dev_idx = device.index if device.index is not None else 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
    allocated_bytes = torch.cuda.memory_allocated(dev_idx)
    reserved_bytes = torch.cuda.memory_reserved(dev_idx)

    return {
        "allocated_gb": round(allocated_bytes / (1024 ** 3), 4),
        "reserved_gb": round(reserved_bytes / (1024 ** 3), 4),
        "free_gb": round(free_bytes / (1024 ** 3), 4),
        "total_gb": round(total_bytes / (1024 ** 3), 4),
    }


__all__ = [
    "resolve_device",
    "set_gpu_memory_limits",
    "check_vram_available",
    "get_provenance_metadata",
    "get_execution_provenance",
    "get_device_provenance",
    "set_cuda_precision_and_determinism",
    "setup_shared_gpu_environment",
    "get_gpu_memory_info",
    "get_git_commit_hash",
]
