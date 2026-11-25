"""
Comprehensive unit & integration tests for RFC-002 (Class Imbalance & F1 Optimization):
- Pillar 1: FocalLoss & Effective-Sample class weighting (Cui et al.)
- Pillar 3: Temperature scaling (NLL on validation logits) + Cost-sensitive F0.5 / constrained_fpr thresholding + length guards
- Pillar 4: Chunked CVaR Top-K pAUC surrogate loss & fine-tuning loop
- Pillar 5: EmbeddingXGBStacker ([z_i || x_i] -> XGBoost) & train_eval_gnn training_metadata contract
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.hdg.data.graph import apply_split_scaling, build_base_graph
from src.hdg.data.splits import DataSplits
from src.hdg.metrics import (
    apply_temperature_scaling,
    choose_optimal_threshold,
    compare_at_matched_fpr,
    evaluate_predictions,
    fit_temperature_scaling,
)
from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.training.stacker import EmbeddingXGBStacker
from src.hdg.training.train import (
    FocalLoss,
    compute_class_balance_weights,
    cvar_topk_pauc_loss,
    train_eval_gnn,
)


def test_focal_loss_and_effective_sample_weights() -> None:
    # Easy negative suppression in FocalLoss
    fl = FocalLoss(alpha=0.75, gamma=2.0, reduction="none")
    easy_neg_logit = torch.tensor([[4.0, -4.0]])
    hard_neg_logit = torch.tensor([[0.1, -0.1]])
    target_zero = torch.tensor([0])
    l_easy = fl(easy_neg_logit, target_zero)
    l_hard = fl(hard_neg_logit, target_zero)
    assert l_easy.item() < l_hard.item() * 0.01

    # Class imbalance weights under 5:1 skew
    y_imb = torch.tensor([0] * 50 + [1] * 10, dtype=torch.long)
    w_cw = compute_class_balance_weights(y_imb, mode="class_weighted")
    assert w_cw is not None
    assert np.isclose(w_cw[1].item(), 5.0, atol=1e-4)

    w_eff = compute_class_balance_weights(y_imb, mode="effective_samples", effective_beta=0.99)
    assert w_eff is not None
    assert w_eff[1].item() > w_eff[0].item()
    assert np.isclose((w_eff[0] + w_eff[1]).item(), 2.0, atol=1e-4)


def test_temperature_scaling_cost_sensitive_f05_and_length_guards() -> None:
    rng = np.random.default_rng(42)
    y_val = np.array([0] * 100 + [1] * 20)
    # Overconfident logits (high magnitude)
    logits_val = np.zeros((120, 2), dtype=np.float64)
    logits_val[:100, 0] = 3.0 + rng.normal(0, 1.0, 100)
    logits_val[:100, 1] = -3.0
    logits_val[100:, 0] = -2.0
    logits_val[100:, 1] = 2.0 + rng.normal(0, 1.0, 20)

    temp = fit_temperature_scaling(logits_val, y_val)
    assert 0.05 <= temp <= 10.0
    probs_cal = apply_temperature_scaling(logits_val, temperature=temp)
    assert probs_cal.shape == (120,)
    assert np.all((probs_cal >= 0.0) & (probs_cal <= 1.0))

    # F0.5 should pick threshold >= F1 under heavy class skew
    probs_skew = np.concatenate([rng.uniform(0.05, 0.55, 200), rng.uniform(0.35, 0.95, 40)])
    y_skew = np.array([0] * 200 + [1] * 40)
    th_f1 = choose_optimal_threshold(y_skew, probs_skew, criterion="f1")
    th_f05 = choose_optimal_threshold(y_skew, probs_skew, criterion="f0.5")
    th_fpr = choose_optimal_threshold(y_skew, probs_skew, criterion="constrained_fpr", max_val_fpr=0.05)
    assert th_f05 >= th_f1
    assert 0.05 <= th_fpr <= 0.95

    # Defensive length guard checks
    with pytest.raises(ValueError, match="Array length mismatch"):
        evaluate_predictions(np.array([0, 1]), np.array([0.2, 0.4, 0.6]))
    with pytest.raises(ValueError, match="Array length mismatch"):
        choose_optimal_threshold(np.array([0, 1]), np.array([0.5]))
    with pytest.raises(ValueError, match="Array length mismatch"):
        compare_at_matched_fpr(np.array([0, 1]), np.array([0.1, 0.2]), np.array([0.1]))


def test_chunked_cvar_topk_pauc_loss_equivalence() -> None:
    torch.manual_seed(7)
    logits = torch.randn(100, 2, requires_grad=True)
    targets = torch.tensor(([1] * 30) + ([0] * 70), dtype=torch.long)

    l_full = cvar_topk_pauc_loss(logits, targets, k_fraction=0.1, margin=0.20, chunk_size=4096)
    l_chunked = cvar_topk_pauc_loss(logits, targets, k_fraction=0.1, margin=0.20, chunk_size=7)
    assert torch.isclose(l_full, l_chunked, atol=1e-6)
    l_chunked.backward()
    assert logits.grad is not None


def test_focal_training_metadata_and_embedding_xgb_stacker() -> None:
    rows = []
    for i in range(40):
        is_mal = (i >= 30)  # natural imbalanced skew (30 benign : 10 mal)
        rows.append({
            "domain": f"item-{i}.{'evil' if is_mal else 'good'}.com",
            "label": 1 if is_mal else 0,
            "source": "ThreatFox" if is_mal else "Umbrella",
            "t": "2023-08-01T00:00:00Z",
            "t_reg": "2023-07-01T00:00:00Z" if is_mal else "2018-01-01T00:00:00Z",
            "no_ip": False,
            "no_rdap": False,
            "no_tls": False,
            "nameservers": [f"ns{i % 2}.prov.net"],
            "registrar": "RegA" if not is_mal else "RegB",
            "leaf_cert_key_cn": f"cert_{i % 3}",
            "resolved_ips": [f"198.51.100.{i % 4}"],
            "ip_records": [{"ip": f"198.51.100.{i % 4}", "asn": 13335 + (i % 2), "asn_org": "TestASN", "country_code": "US", "prefix_len": 24}],
            "ns_ip_pairs": [(f"ns{i % 2}.prov.net", f"198.51.100.{i % 4}")],
        })
    df = pd.DataFrame(rows)
    base = build_base_graph(df)
    splits = DataSplits(
        train_indices=np.array(list(range(0, 20)) + list(range(30, 36))),
        val_indices=np.array(list(range(20, 25)) + list(range(36, 38))),
        test_indices=np.array(list(range(25, 30)) + list(range(38, 40))),
        split_type="random",
    )
    scaled, _ = apply_split_scaling(base, splits.train_indices, val_indices=splits.val_indices, test_indices=splits.test_indices)
    in_ch = {nt: scaled[nt].x.shape[1] for nt in scaled.node_types}
    model = HeteroGNN(
        metadata=scaled.metadata(),
        in_channels_dict=in_ch,
        hidden_dim=16,
        out_dim=8,
        num_layers=2,
        variant="sage",
        use_char_cnn=True,
        char_cnn_dim=8,
        use_completeness_gate=True,
    )

    test_eval, _, _, trained_m = train_eval_gnn(
        model=model,
        data=scaled,
        splits=splits,
        df=df,
        epochs=4,
        patience=2,
        min_epochs=1,
        bootstrap_samples=10,
        loss_type="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        threshold_criterion="f0.5",
        use_temperature_scaling=True,
        seed=42,
    )
    meta = test_eval["training_metadata"]
    assert meta["loss_type"] == "focal"
    assert meta["focal_gamma"] == 2.0
    assert meta["threshold_criterion"] == "f0.5"
    assert "temperature" in meta and meta["temperature"] > 0.0
    assert "calibrated_threshold" in meta

    # Verify Hybrid EmbeddingXGBStacker (Pillar 5)
    trained_m.eval()
    ew = {et: scaled[et].edge_weight for et in scaled.edge_types if hasattr(scaled[et], "edge_weight")}
    with torch.no_grad():
        _, penult = trained_m.get_penultimate_embeddings(
            scaled.x_dict, scaled.edge_index_dict, edge_weight_dict=ew, char_bytes=scaled["domain"].char_bytes
        )
    assert penult.shape == (40, 8)

    stacker = EmbeddingXGBStacker(max_depth=3, n_estimators=10, random_state=42)
    fit_info = stacker.fit(
        embeddings=penult.numpy(),
        domain_features=scaled["domain"].x.numpy(),
        y_all=scaled["domain"].y.numpy(),
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        threshold_criterion="f0.5",
    )
    assert fit_info["feature_dim"] == 8 + scaled["domain"].x.shape[1]
    st_metrics, _, st_probs = stacker.evaluate(
        embeddings=penult.numpy(),
        domain_features=scaled["domain"].x.numpy(),
        y_all=scaled["domain"].y.numpy(),
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
    )
    assert len(st_probs) == len(splits.test_indices)
    assert "roc_auc" in st_metrics
