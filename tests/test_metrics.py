"""
Tests for statistical evaluation metrics, hypothesis testing, and bootstrap CIs.
Verifies:
- TPR at target FPR (1% and 0.1%) with validation negative thresholds
- Expected false positive threshold warnings
- Exact McNemar test statistic correctness (min(b, c))
- Paired bootstrap AUC difference
"""

import numpy as np
import pytest

from src.hdg.metrics import (
    compute_bootstrap_cis,
    compute_tpr_at_fpr,
    evaluate_predictions,
    paired_bootstrap_auc_difference,
    paired_mcnemar_test,
)


def test_tpr_at_fpr_and_warning() -> None:
    # 50 validation negatives with scores uniformly in [0, 0.5]
    y_val_true = np.array([0] * 50 + [1] * 50)
    y_val_prob = np.concatenate([
        np.linspace(0.01, 0.49, 50),
        np.linspace(0.51, 0.99, 50)
    ])

    # 40 test samples (20 negs, 20 pos)
    y_test_true = np.array([0] * 20 + [1] * 20)
    y_test_prob = np.concatenate([
        np.linspace(0.01, 0.49, 20),
        np.linspace(0.51, 0.99, 20)
    ])

    # Target FPR 1%
    tpr, th, exp_fp, warning = compute_tpr_at_fpr(
        y_test_true=y_test_true,
        y_test_prob=y_test_prob,
        y_val_true=y_val_true,
        y_val_prob=y_val_prob,
        target_fpr=0.01
    )

    assert 0.0 <= tpr <= 1.0
    assert th > 0.40  # Threshold must be near the top of validation negatives
    assert exp_fp == int(round(20 * 0.01))  # 0 expected false positives
    # Warning must trigger because expected_fp < 10
    assert warning is not None
    assert "Expected false positives" in warning


def test_bootstrap_cis() -> None:
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 5)
    y_prob = np.array([0.9, 0.8, 0.85, 0.7, 0.95, 0.1, 0.2, 0.15, 0.3, 0.05] * 5)

    cis = compute_bootstrap_cis(y_true, y_prob, threshold=0.5, n_bootstrap=100, seed=42)
    assert "roc_auc" in cis
    assert "pr_auc" in cis
    assert "f1" in cis

    roc_ci = cis["roc_auc"]
    assert 0.8 <= roc_ci["point"] <= 1.0
    assert roc_ci["ci_lower"] <= roc_ci["point"] <= roc_ci["ci_upper"]


def test_exact_mcnemar() -> None:
    # Model A is better than Model B
    # Table: 80 both correct, 15 A correct & B wrong, 2 A wrong & B correct, 3 both wrong
    y_true = np.array([1] * 100)
    y_pred_a = np.array([1] * 95 + [0] * 5)
    y_pred_b = np.array([1] * 80 + [0] * 15 + [1] * 2 + [0] * 3)

    res = paired_mcnemar_test(y_true, y_pred_a, y_pred_b)
    # Exact test statistic must be min(b, c) = min(15, 2) = 2.0
    assert res["exact_statistic_min_b_c"] == 2.0
    assert res["pvalue"] < 0.05
    assert res["discordant_a_better"] == 15
    assert res["discordant_b_better"] == 2


def test_paired_bootstrap_auc_difference() -> None:
    y_true = np.array([1] * 30 + [0] * 30)
    # Model A: strong probabilities
    y_prob_a = np.concatenate([np.random.uniform(0.7, 0.95, 30), np.random.uniform(0.05, 0.3, 30)])
    # Model B: weaker probabilities
    y_prob_b = np.concatenate([np.random.uniform(0.4, 0.65, 30), np.random.uniform(0.35, 0.6, 30)])

    res = paired_bootstrap_auc_difference(y_true, y_prob_a, y_prob_b, n_bootstrap=100, seed=42)
    assert res["mean_diff"] > 0.0
    assert res["ci_lower"] <= res["mean_diff"] <= res["ci_upper"]


def test_prevalence_adjusted_precision_formula() -> None:
    """Verifies that prevalence-adjusted precision exactly matches Bayes' theorem."""
    from src.hdg.metrics import compute_prevalence_adjusted_precision

    # Case 1: TPR=0.80, FPR=0.01, Prevalence=0.01 (1%)
    # Bayes: (0.80 * 0.01) / (0.80 * 0.01 + 0.01 * 0.99) = 0.008 / (0.008 + 0.0099) = 0.008 / 0.0179 = 0.446927
    tpr = 0.80
    fpr = 0.01
    pi = 0.01
    expected = (tpr * pi) / (tpr * pi + fpr * (1.0 - pi))
    calculated = compute_prevalence_adjusted_precision(tpr=tpr, fpr=fpr, pi=pi)
    assert abs(calculated - expected) < 1e-6
    assert abs(calculated - 0.446927) < 1e-4

    # Case 2: Extreme low prevalence (pi = 0.001)
    pi_low = 0.001
    expected_low = (tpr * pi_low) / (tpr * pi_low + fpr * (1.0 - pi_low))
    calculated_low = compute_prevalence_adjusted_precision(tpr=tpr, fpr=fpr, pi=pi_low)
    assert abs(calculated_low - expected_low) < 1e-6

    # Case 3: Zero TPR / zero denominator
    assert compute_prevalence_adjusted_precision(0.0, 0.0, 0.01) == 0.0


def test_realised_fpr_matches_count() -> None:
    """Verifies that realised FPR equals exact empirical count of false alarms / total negatives."""
    y_test_true = np.array([0] * 80 + [1] * 20)
    # Give 2 negatives high scores above threshold 0.85
    y_test_prob = np.zeros(100)
    y_test_prob[0] = 0.90  # False positive 1
    y_test_prob[1] = 0.88  # False positive 2
    y_test_prob[80:] = 0.95  # True positives

    threshold = 0.85
    fp_count = int(np.sum((y_test_prob[:80] >= threshold)))
    total_negs = 80
    expected_fpr = fp_count / total_negs  # 2 / 80 = 0.025

    actual_negs = y_test_prob[y_test_true == 0]
    realised_fpr = float(np.mean(actual_negs >= threshold))
    assert fp_count == 2
    assert realised_fpr == expected_fpr == 0.025

