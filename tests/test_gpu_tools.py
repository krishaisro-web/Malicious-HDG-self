"""
Unit tests for GPU pre-flight verification and device parity tooling.
"""

import json
import subprocess
import sys
from pathlib import Path
import unittest
import torch

from tools.verify_gpu_graph import run_preflight_checks


class TestGpuTools(unittest.TestCase):
    def test_run_preflight_checks_on_cpu(self):
        """Preflight checks must pass on CPU and emit valid JSON."""
        tmp_dir = Path("results/mac/test_preflight_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        res = run_preflight_checks(
            device_str="cpu",
            out_dir=str(tmp_dir),
            graph_path="data/processed/real/heterodata.pt",
            mem_fraction=0.35,
            min_free_vram_gb=8.0,
        )

        self.assertTrue(res["passed_all"])
        self.assertEqual(res["checks"]["device_resolution"]["status"], "PASS")
        self.assertEqual(res["checks"]["strict_fp32"]["status"], "PASS")
        self.assertEqual(res["checks"]["synthetic_gnn_execution"]["status"], "PASS")

        json_file = tmp_dir / "preflight_verification.json"
        self.assertTrue(json_file.exists())
        with open(json_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["provenance"]["device_type"], "cpu")

    def test_latency_eval_rejects_cuda(self):
        """Constraint #7: pipeline/07_latency_eval.py must strictly refuse --device cuda."""
        cmd = [
            sys.executable,
            "pipeline/07_latency_eval.py",
            "--device",
            "cuda",
            "--smoke",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LATENCY ISOLATION SAFEGUARD", result.stderr)

    def test_device_parity_tool_runs_on_cpu(self):
        """Seed 42 device parity tool must execute cleanly and emit valid JSON."""
        from tools.verify_device_parity import run_device_parity_check
        tmp_dir = Path("results/mac/test_parity_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        res = run_device_parity_check(
            seed=42,
            config_path="configs/active_config.yaml",
            out_dir=str(tmp_dir),
            max_delta=0.002,
        )

        self.assertTrue(res["parity_verified"])
        self.assertIn("cpu_reference", res)
        self.assertEqual(res["cpu_reference"]["device"], "cpu")

        json_file = tmp_dir / "device_parity.json"
        self.assertTrue(json_file.exists())
        with open(json_file, "r") as f:
            data = json.load(f)
        self.assertTrue(data["parity_verified"])


if __name__ == "__main__":
    unittest.main()
