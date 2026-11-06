"""
Adversarial stress-testing suite for device resolution, CLI constraints,
CPU-only latency safeguards, FP32 precision/determinism, and atomic attack tensor placement.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest import mock
import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData

from src.hdg.device import (
    resolve_device,
    set_cuda_precision_and_determinism,
    setup_shared_gpu_environment,
    get_provenance_metadata,
    check_vram_available,
    set_gpu_memory_limits,
)
from src.hdg.data.splits import DataSplits
from src.hdg.eval.attack import generate_structural_attack, evaluate_attacked_model
from src.hdg.models.hetero_gnn import HeteroGNN


class TestDeviceResolutionAdversarial:
    """Stress tests for device resolution boundary conditions."""

    def test_invalid_device_strings_raise_value_error(self):
        """Random invalid non-cuda strings must raise ValueError."""
        invalid_strings = ["foo", "bar", "tpu", "rocm", "gpu", "None", "cpu:0"]
        for bad in invalid_strings:
            with pytest.raises(ValueError) as excinfo:
                resolve_device(bad)
            assert "Unsupported device" in str(excinfo.value)

    def test_empty_and_whitespace_default_to_cpu(self):
        """Empty or whitespace strings must safely default to CPU."""
        with mock.patch.dict(os.environ, {}, clear=True):
            for empty_val in ["", "   ", "\t", "\n"]:
                dev = resolve_device(empty_val)
                assert dev.type == "cpu"

    def test_none_defaults_to_cpu(self):
        """None must default to CPU when HDG_DEVICE is unset."""
        with mock.patch.dict(os.environ, {}, clear=True):
            dev = resolve_device(None)
            assert dev.type == "cpu"

    def test_cuda_when_cuda_unavailable_raises_runtime_error(self):
        """When CUDA is unavailable, any cuda request must raise RuntimeError."""
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            for cuda_str in ["cuda", "cuda:0", "cuda:99", "CUDA", "CUDA:1", "cuda_0"]:
                with pytest.raises(RuntimeError) as excinfo:
                    resolve_device(cuda_str)
                assert "torch.cuda.is_available() is False" in str(excinfo.value)

    def test_malformed_cuda_strings_raise_runtime_error(self):
        """Malformed CUDA strings (cuda:foo, cuda:-1) must raise RuntimeError."""
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            for bad in ["cuda:foo", "cuda:-1", "cuda::0", "cuda:"]:
                with pytest.raises(RuntimeError) as excinfo:
                    resolve_device(bad)
                assert "Invalid device string" in str(excinfo.value)

    def test_mps_device_behavior(self):
        """MPS device is supported on Apple Silicon or rejected if unavailable."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = resolve_device("mps")
            assert dev.type == "mps"
        else:
            with pytest.raises(RuntimeError) as excinfo:
                resolve_device("mps")
            assert "torch.backends.mps.is_available() is False" in str(excinfo.value)

    def test_non_string_arguments_raise_attribute_error(self):
        """Passing non-string inputs (like integers or torch.device) raises AttributeError."""
        for non_str in [123, True, torch.device("cpu")]:
            with pytest.raises(AttributeError):
                resolve_device(non_str)


class TestCliDeviceConstraints:
    """Stress tests for CLI script device arguments and defaults."""

    @pytest.mark.parametrize(
        "script_path",
        [
            "pipeline/04_train_gnn.py",
            "pipeline/05_ablation.py",
            "pipeline/06_attack_eval.py",
            "pipeline/07_latency_eval.py",
            "tools/verify_gpu_graph.py",
        ],
    )
    def test_cli_help_documents_device_and_defaults_to_cpu(self, script_path):
        """Every CLI script must expose --device and default to cpu."""
        res = subprocess.run(
            [sys.executable, script_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert res.returncode == 0
        assert "--device" in res.stdout

    def test_cli_ignores_hdg_device_when_omitted_due_to_argparse_default(self):
        """
        Adversarial finding: because scripts configure default='cpu' in argparse,
        export HDG_DEVICE=cuda has no effect on CLI entry points when --device is omitted.
        """
        env = os.environ.copy()
        env["HDG_DEVICE"] = "cuda"

        # Explicit --device cuda fails on Mac at line 78 of 04_train_gnn.py
        res_explicit = subprocess.run(
            [sys.executable, "pipeline/04_train_gnn.py", "--device", "cuda", "--smoke"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert res_explicit.returncode != 0
        assert "CUDA device requested" in res_explicit.stderr

        # Omitted --device with HDG_DEVICE=cuda does NOT trigger CUDA device resolution; defaults to cpu
        res_env = subprocess.run(
            [sys.executable, "pipeline/04_train_gnn.py", "--smoke", "--single-seed", "42"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert "CUDA device requested" not in res_env.stderr


class TestLatencyEvalSafeguard:
    """Stress tests for Latency Eval strictly CPU-only safeguard."""

    @pytest.mark.parametrize(
        "forbidden_device",
        [
            "cuda",
            "cuda:0",
            "cuda:1",
            "CUDA",
            "CuDa:0",
            "mps",
            "MPS",
            "auto",
            "gpu",
            "",
            "   ",
        ],
    )
    def test_latency_eval_rejects_non_cpu_devices(self, forbidden_device):
        """pipeline/07_latency_eval.py must strictly reject any non-cpu device string."""
        cmd = [
            sys.executable,
            "pipeline/07_latency_eval.py",
            "--device",
            forbidden_device,
            "--smoke",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode != 0
        assert "LATENCY ISOLATION SAFEGUARD" in res.stderr

    @pytest.mark.parametrize("valid_cpu", ["cpu", "CPU", " cpu "])
    def test_latency_eval_accepts_cpu_variants(self, valid_cpu):
        """pipeline/07_latency_eval.py passes safeguard for valid cpu strings."""
        cmd = [
            sys.executable,
            "pipeline/07_latency_eval.py",
            "--device",
            valid_cpu,
            "--smoke",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Safeguard must NOT be in stderr
        assert "LATENCY ISOLATION SAFEGUARD" not in res.stderr


class TestFP32PrecisionAndDeterminism:
    """Stress tests for FP32 precision and determinism enforcement."""

    def test_set_cuda_precision_and_determinism_resets_tf32(self):
        """Calling set_cuda_precision_and_determinism disables allow_tf32."""
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        set_cuda_precision_and_determinism()

        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            assert torch.backends.cuda.matmul.allow_tf32 is False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True

    def test_setup_shared_gpu_environment_safeguards(self):
        """setup_shared_gpu_environment enforces fraction cap and aborts on low VRAM."""
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            # 1. Low VRAM aborts
            with mock.patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(4 * (1024**3), 80 * (1024**3)),
            ):
                with pytest.raises(RuntimeError) as excinfo:
                    setup_shared_gpu_environment(dev_idx=0, min_free_vram_gb=8.0)
                assert "Insufficient VRAM" in str(excinfo.value)

            # 2. Sufficient VRAM configures fraction cap
            with mock.patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(20 * (1024**3), 80 * (1024**3)),
            ):
                with mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"):
                    with mock.patch.object(torch.cuda, "set_per_process_memory_fraction") as mock_frac:
                        res = setup_shared_gpu_environment(
                            dev_idx=0, memory_fraction=0.35, min_free_vram_gb=8.0
                        )
                        mock_frac.assert_called_once_with(0.35, 0)
                        assert res["memory_fraction_cap"] == 0.35
                        assert res["free_vram_gb"] == 20.0

    def test_get_provenance_metadata_structure(self):
        """get_provenance_metadata contains all required provenance fields."""
        prov = get_provenance_metadata(torch.device("cpu"))
        required_keys = [
            "device",
            "device_type",
            "torch_version",
            "cuda_available",
            "cuda_version",
            "git_commit",
            "hostname",
            "platform",
            "precision",
            "tf32_matmul_enabled",
            "tf32_cudnn_enabled",
        ]
        for k in required_keys:
            assert k in prov
        assert prov["precision"] == "FP32"
        assert prov["tf32_matmul_enabled"] is False
        assert prov["tf32_cudnn_enabled"] is False


class TestAtomicAttackTensorPlacement:
    """Stress tests for atomic attack tensor staging."""

    @pytest.fixture
    def synthetic_hetero_data(self):
        num_domains = 50
        data = HeteroData()
        data["domain"].x = torch.randn(num_domains, 12, dtype=torch.float32)
        data["domain"].y = torch.tensor([0 if i % 2 == 0 else 1 for i in range(num_domains)], dtype=torch.long)
        num_ips = 20
        data["ip"].num_nodes = num_ips
        data["ip"].x = torch.randn(num_ips, 1, dtype=torch.float32)

        src_d = torch.randint(0, num_domains, (100,))
        dst_ip = torch.randint(0, num_ips, (100,))
        data["domain", "resolves_to", "ip"].edge_index = torch.stack([src_d, dst_ip], dim=0)
        data["ip", "rev_resolves_to", "domain"].edge_index = torch.stack([dst_ip, src_d], dim=0)
        return data

    def test_generate_structural_attack_preserves_clean_data_device(self, synthetic_hetero_data):
        """
        Verifies that HeteroData is deep-copied before staging to CPU so clean_data
        is not mutated in-place when clean_data is on GPU/MPS.
        Subsequent attack iterations receive clean_data intact on its original device.
        """
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            pytest.skip("MPS not available on this host")

        clean_data = synthetic_hetero_data.to("mps")
        splits = DataSplits(
            train_indices=np.arange(0, 30),
            val_indices=np.arange(30, 40),
            test_indices=np.arange(40, 50),
            split_type="test",
        )

        assert clean_data["domain"].x.device.type == "mps"

        # Budget 1
        atk1 = generate_structural_attack(clean_data, splits, budget=1)
        assert atk1["domain"].x.device.type == "mps"

        # Immutability check: clean_data remains on mps (NOT mutated to CPU)
        assert clean_data["domain"].x.device.type == "mps", "clean_data device unexpectedly mutated to CPU"

        # Budget 2: Because clean_data was preserved on mps, atk2 is also returned on mps!
        atk2 = generate_structural_attack(clean_data, splits, budget=2)
        assert atk2["domain"].x.device.type == "mps", "atk2 was not on mps after second attack iteration"

    def test_generate_structural_attack_on_cpu(self, synthetic_hetero_data):
        """generate_structural_attack on CPU works cleanly without device side-effects."""
        splits = DataSplits(
            train_indices=np.arange(0, 30),
            val_indices=np.arange(30, 40),
            test_indices=np.arange(40, 50),
            split_type="test",
        )
        atk = generate_structural_attack(synthetic_hetero_data, splits, budget=2)
        assert atk["domain"].x.device.type == "cpu"
        assert atk["domain", "resolves_to", "ip"].edge_index.device.type == "cpu"

    def test_generate_structural_attack_edge_cases(self, synthetic_hetero_data):
        """generate_structural_attack handles edge cases (empty malicious test or train benign)."""
        splits = DataSplits(
            train_indices=np.arange(0, 30),
            val_indices=np.arange(30, 40),
            test_indices=np.arange(40, 50),
            split_type="test",
        )

        # Edge case: all test nodes are benign
        data_all_benign = synthetic_hetero_data.clone()
        data_all_benign["domain"].y[splits.test_indices] = 0
        atk_res = generate_structural_attack(data_all_benign, splits, budget=2)
        assert atk_res["domain"].x.device.type == "cpu"
