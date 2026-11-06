"""
Unit tests for centralized device resolution and hardware provenance (src/hdg/device.py).
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import torch

from src.hdg.device import (
    resolve_device,
    set_gpu_memory_limits,
    check_vram_available,
    get_provenance_metadata,
    get_execution_provenance,
    get_device_provenance,
    set_cuda_precision_and_determinism,
    get_gpu_memory_info,
    setup_shared_gpu_environment,
)


class TestDeviceResolution(unittest.TestCase):
    def test_default_is_cpu(self):
        """Constraint #2: Centralized resolver must default strictly to CPU."""
        with patch.dict(os.environ, {}, clear=True):
            dev = resolve_device(None)
            self.assertEqual(dev.type, "cpu")

            dev_empty = resolve_device("")
            self.assertEqual(dev_empty.type, "cpu")

            dev_no_args = resolve_device()
            self.assertEqual(dev_no_args.type, "cpu")

    def test_env_var_override(self):
        with patch.dict(os.environ, {"HDG_DEVICE": "cpu"}):
            dev = resolve_device(None)
            self.assertEqual(dev.type, "cpu")

        with patch.dict(os.environ, {"HDG_DEVICE": "cuda"}):
            with patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaises(RuntimeError):
                    resolve_device(None)

    def test_explicit_cpu(self):
        dev = resolve_device("cpu")
        self.assertEqual(dev.type, "cpu")

        dev_kwarg = resolve_device(device_arg="cpu")
        self.assertEqual(dev_kwarg.type, "cpu")

        dev_str_kwarg = resolve_device(device_str="cpu")
        self.assertEqual(dev_str_kwarg.type, "cpu")

    def test_invalid_device_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_device("tpu_v4")

    def test_cuda_unavailable_raises_runtime_error(self):
        """When CUDA is not present, requesting cuda must fail loudly."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                resolve_device("cuda")
            self.assertIn("torch.cuda.is_available() is False", str(ctx.exception))

    def test_cuda_enforces_fp32_and_determinism(self):
        """Constraint #3 & #5: CUDA resolution must disable TF32 and set determinism."""
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch("torch.device") as mock_device:
                mock_device.return_value = torch.device("cpu")
                _ = resolve_device("cuda:0")

                # Verify TF32 was disabled
                if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_get_provenance_metadata_keys(self):
        """Constraint #2: Provenance must contain auditable metadata fields."""
        dev = torch.device("cpu")
        prov = get_provenance_metadata(dev)
        expected_keys = [
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
        for key in expected_keys:
            self.assertIn(key, prov, f"Missing key '{key}' in provenance dict")
        self.assertEqual(prov["precision"], "FP32")
        self.assertEqual(prov["device_type"], "cpu")
        self.assertEqual(prov["device"], "cpu")

    def test_get_execution_provenance_alias(self):
        dev = torch.device("cpu")
        prov1 = get_provenance_metadata(dev)
        prov2 = get_execution_provenance(dev)
        prov3 = get_device_provenance(dev)
        self.assertEqual(prov1, prov2)
        self.assertEqual(prov1, prov3)

    def test_set_cuda_precision_and_determinism(self):
        # Should execute safely without error
        set_cuda_precision_and_determinism()
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_get_gpu_memory_info_on_cpu(self):
        info = get_gpu_memory_info(torch.device("cpu"))
        self.assertEqual(info["allocated_gb"], 0.0)
        self.assertEqual(info["reserved_gb"], 0.0)
        self.assertEqual(info["free_gb"], 0.0)
        self.assertEqual(info["total_gb"], 0.0)

    def test_set_gpu_memory_limits_on_cpu_or_unavailable(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            # Should safely return without error
            set_gpu_memory_limits(0.35, torch.device("cpu"))

        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "set_per_process_memory_fraction") as mock_set:
                set_gpu_memory_limits(0.35, torch.device("cpu"))
                mock_set.assert_not_called()

    def test_set_gpu_memory_limits_on_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "set_per_process_memory_fraction") as mock_set:
                set_gpu_memory_limits(0.35, torch.device("cuda:0"))
                mock_set.assert_called_once_with(0.35, 0)

    def test_check_vram_available(self):
        # 1. When CUDA is unavailable, returns False
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.assertFalse(check_vram_available(8.0))
            self.assertFalse(check_vram_available(8.0, torch.device("cpu")))

        # 2. When CUDA is available, returns True if free >= threshold
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(10 * (1024 ** 3), 80 * (1024 ** 3)),
            ):
                self.assertTrue(check_vram_available(8.0, torch.device("cuda:0")))
                self.assertFalse(check_vram_available(12.0, torch.device("cuda:0")))

    def test_setup_shared_gpu_environment(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                setup_shared_gpu_environment(dev_idx=0)

        # Insufficient VRAM aborts
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(4 * (1024 ** 3), 80 * (1024 ** 3)),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    setup_shared_gpu_environment(dev_idx=0, min_free_vram_gb=8.0)
                self.assertIn("Insufficient VRAM", str(ctx.exception))

        # Sufficient VRAM succeeds
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(20 * (1024 ** 3), 80 * (1024 ** 3)),
            ):
                with patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"):
                    with patch.object(torch.cuda, "set_per_process_memory_fraction") as mock_frac:
                        result = setup_shared_gpu_environment(
                            dev_idx=0, memory_fraction=0.35, min_free_vram_gb=8.0
                        )
                        mock_frac.assert_called_once_with(0.35, 0)
                        self.assertEqual(result["device_index"], 0)
                        self.assertEqual(result["memory_fraction_cap"], 0.35)
                        self.assertEqual(result["free_vram_gb"], 20.0)


if __name__ == "__main__":
    unittest.main()
