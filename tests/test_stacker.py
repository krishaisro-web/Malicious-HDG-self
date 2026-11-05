"""
Unit tests for Post-Hoc Stacker with GBA features and pre-committed stop rule.
"""

import numpy as np
import pytest
from torch_geometric.data import HeteroData

from src.hdg.training.stacker import (
    compute_smoothed_gba_scores,
    extract_gba_features,
    PostHocStacker,
    evaluate_stacker_against_stop_rule
)


def test_smoothed_gba_scores_and_feature_extraction(minimal_synthetic_heterodata: HeteroData) -> None:
    data = minimal_synthetic_heterodata
    train_idx = np.array([0, 1, 2, 3])
    train_labels = np.array([0, 1, 0, 1])

    # 1. Compute GBA scores
    gba = compute_smoothed_gba_scores(data, train_idx, train_labels, smoothing_a=5.0)
    assert "ip" in gba
    assert "nameserver" in gba
    # Scores must be bounded in [0, 1]
    for nt, scores in gba.items():
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    # 2. Extract GBA features for all domains
    domain_idx = np.arange(10)
    feats = extract_gba_features(data, domain_idx, gba)
    assert feats.shape[0] == 10
    assert feats.shape[1] > 0
    assert np.all(feats >= 0.0) and np.all(feats <= 1.0)


def test_post_hoc_stacker_fit_predict() -> None:
    rng = np.random.default_rng(42)
    n_val = 100
    y_val = rng.integers(0, 2, size=n_val)

    # Base model probabilities: SAGE, ATTN, XGBoost, TF-IDF
    base_probs = np.column_stack([
        rng.uniform(0.1, 0.9, size=n_val),
        rng.uniform(0.1, 0.9, size=n_val),
        rng.uniform(0.1, 0.9, size=n_val),
        rng.uniform(0.1, 0.9, size=n_val)
    ])
    gba_feats = rng.uniform(0.2, 0.8, size=(n_val, 8))

    stacker = PostHocStacker(c_reg=1.0, random_state=42)
    stacker.fit_cross_validated(base_probs, gba_feats, y_val, n_splits=3)
    assert stacker.is_fitted

    test_preds = stacker.predict_proba(base_probs, gba_feats)
    assert len(test_preds) == n_val
    assert np.all(test_preds >= 0.0) and np.all(test_preds <= 1.0)


def test_stop_rule_evaluation() -> None:
    rng = np.random.default_rng(42)
    y_test = np.array([0] * 500 + [1] * 250)
    y_val = np.array([0] * 500 + [1] * 250)

    sage_val = np.concatenate([rng.uniform(0.01, 0.3, 500), rng.uniform(0.4, 0.9, 250)])
    sage_test = np.concatenate([rng.uniform(0.01, 0.3, 500), rng.uniform(0.4, 0.9, 250)])

    # Case 1: Modest improvement below +3.0 pp
    stacker_val = sage_val + rng.normal(0.0, 0.01, len(sage_val))
    stacker_test = sage_test + rng.normal(0.0, 0.01, len(sage_test))

    res = evaluate_stacker_against_stop_rule(
        y_test=y_test,
        sage_probs=sage_test,
        stacker_probs=stacker_test,
        y_val=y_val,
        sage_val_probs=sage_val,
        stacker_val_probs=stacker_val,
        stop_rule_gain_threshold=0.030,
        seed=42
    )
    assert "status" in res
    assert "delta_tpr" in res
    assert "interpretation" in res
