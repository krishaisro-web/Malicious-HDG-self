"""
Unit tests for HP Thin Client configuration, thread capping, and memory safety.
"""

import os
from pathlib import Path
import pytest
import torch

from src.hdg.config import (
    load_config,
    init_thread_pool,
    is_thin_client_environment,
    detect_system_hardware,
    generate_dynamic_config,
)


def test_hp_thin_client_config_loading():
    """Verifies that hp_thin_client.yaml loads correctly with constrained hardware limits."""
    cfg = load_config("configs/hp_thin_client.yaml")
    
    assert cfg["hardware"]["profile"] == "hp_thin_client"
    assert cfg["hardware"]["num_threads"] == 2
    assert cfg["graph"]["max_domains"] == 30000
    assert cfg["parsing"]["max_per_class"] == 15000
    assert cfg["model"]["training"]["batch_size"] == 256
    assert cfg["model"]["training"]["epochs"] == 50
    assert cfg["evaluation"]["bootstrap_samples"] == 200
    assert len(cfg["randomness"]["evaluation_seeds"]) == 3


def test_env_profile_thin_client(monkeypatch):
    """Tests that HDG_PROFILE=thin_client correctly routes load_config to hp_thin_client.yaml."""
    monkeypatch.setenv("HDG_PROFILE", "thin_client")
    cfg = load_config()
    assert cfg["hardware"]["profile"] == "hp_thin_client"
    assert cfg["graph"]["max_domains"] == 30000


def test_thread_pool_capping_thin_client(monkeypatch):
    """Verifies that init_thread_pool caps threads properly and exports OpenMP/MKL variables."""
    cfg = {"hardware": {"num_threads": 2}}
    monkeypatch.delenv("HDG_THREADS", raising=False)
    
    threads = init_thread_pool(cfg)
    assert threads == 2
    assert torch.get_num_threads() == 2
    assert os.environ.get("OMP_NUM_THREADS") == "2"
    assert os.environ.get("MKL_NUM_THREADS") == "2"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "2"


def test_is_thin_client_environment_detection(monkeypatch):
    """Tests thin client detection helper with environment variable."""
    monkeypatch.setenv("HDG_PROFILE", "hp_thin_client")
    assert is_thin_client_environment() is True


def test_detect_system_hardware():
    """Tests that hardware prober extracts valid system specifications."""
    hw = detect_system_hardware()
    assert isinstance(hw["cpu_cores"], int)
    assert hw["cpu_cores"] >= 1
    assert isinstance(hw["total_ram_gb"], float)
    assert hw["total_ram_gb"] > 0.0


def test_generate_dynamic_config_enterprise(tmp_path):
    """Verifies dynamic configuration tuning for enterprise servers (e.g. 245GB RAM ISRO compute server)."""
    target_yaml = tmp_path / "enterprise_active.yaml"
    hw_fake = {
        "cpu_cores": 32,
        "total_ram_gb": 245.0,
        "avail_ram_gb": 220.0,
        "total_swap_gb": 16.0,
        "free_swap_gb": 16.0,
        "cuda_available": False,
        "platform": "Linux ISRO-Compute",
    }
    out_path = generate_dynamic_config(
        output_path=target_yaml,
        hw_override=hw_fake,
        verbose=False
    )
    assert out_path.exists()
    cfg = load_config(str(out_path))

    assert cfg["hardware"]["profile"] == "enterprise_server"
    assert cfg["hardware"]["num_threads"] == 32
    assert cfg["graph"]["max_domains"] == 100000
    assert cfg["parsing"]["max_per_class"] is None  # Full dataset
    assert cfg["model"]["training"]["batch_size"] == 1024
    assert cfg["model"]["training"]["epochs"] == 200
    assert len(cfg["randomness"]["evaluation_seeds"]) == 5


def test_generate_dynamic_config_thin_client(tmp_path):
    """Verifies dynamic configuration tuning for standard HP Thin Clients (e.g. 8GB RAM)."""
    target_yaml = tmp_path / "thin_active.yaml"
    hw_fake = {
        "cpu_cores": 4,
        "total_ram_gb": 8.0,
        "avail_ram_gb": 6.0,
        "total_swap_gb": 4.0,
        "free_swap_gb": 4.0,
        "cuda_available": False,
        "platform": "Linux ISRO-ThinClient",
    }
    out_path = generate_dynamic_config(
        output_path=target_yaml,
        hw_override=hw_fake,
        verbose=False
    )
    assert out_path.exists()
    cfg = load_config(str(out_path))

    assert cfg["hardware"]["profile"] == "hp_thin_client_standard"
    assert cfg["hardware"]["num_threads"] == 2
    assert cfg["graph"]["max_domains"] == 30000
    assert cfg["parsing"]["max_per_class"] == 15000
    assert cfg["model"]["training"]["batch_size"] == 256
    assert cfg["model"]["training"]["epochs"] == 50
    assert len(cfg["randomness"]["evaluation_seeds"]) == 3


def test_training_checkpoint_and_resume(tmp_path):
    """Verifies that train_eval_gnn saves intermediate state and can resume from it."""
    from torch_geometric.data import HeteroData
    from src.hdg.data.splits import DataSplits
    from src.hdg.models.hetero_gnn import HeteroGNN
    from src.hdg.training.train import train_eval_gnn
    import pandas as pd
    import numpy as np

    data = HeteroData()
    data["domain"].x = torch.randn(10, 8)
    data["domain"].y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    data["ip"].x = torch.randn(5, 8)
    data["domain", "resolves_to", "ip"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    data["ip", "rev_resolves_to", "domain"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

    splits = DataSplits(
        train_indices=np.array([0, 1, 2, 3, 4, 5]),
        val_indices=np.array([6, 7]),
        test_indices=np.array([8, 9]),
        split_type="random"
    )
    df = pd.DataFrame({
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "no_ip": [False] * 10,
        "source": ["src1"] * 10,
        "family": ["fam1"] * 10
    })



    ckpt_file = tmp_path / "model.pt"
    log_file = tmp_path / "model.jsonl"

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict={"domain": 8, "ip": 8},
        hidden_dim=16,
        out_dim=8,
        variant="sage"
    )

    eval_res, _, _, _ = train_eval_gnn(
        model=model,
        data=data,
        splits=splits,
        df=df,
        epochs=2,
        checkpoint_path=ckpt_file,
        log_file=log_file,
        bootstrap_samples=10,
        resume=True
    )
    assert ckpt_file.exists()
    assert log_file.exists()
    assert "roc_auc" in eval_res


