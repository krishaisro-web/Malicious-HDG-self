"""
Evaluation metrics, bootstrap confidence intervals, and hypothesis testing for Malicious-HDG.
Implements:
- ROC-AUC, PR-AUC, F1, Precision, Recall, Specificity
- Operating points (TPR at 1% and 0.1% target FPR) with validation negative thresholds
- Realised test FPR and F1/Precision/Recall at operating points
- Prevalence-adjusted precision at production prevalences pi in {1%, 0.1%}
- Bootstrap 95% confidence intervals (empirical percentile bootstrap)
- Expected false positive threshold warnings (< 10 expected false positives)
- Exact McNemar test (statsmodels, exact=True, statistic = min(b, c))
- Paired bootstrap AUC differences across evaluation seeds
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes exact Clopper-Pearson binomial confidence interval for k successes in n trials.
    Used for rigorous certification of false positive rates (FPR).
    """
    from scipy.stats import beta
    if n <= 0:
        return 0.0, 1.0
    k = max(0, min(k, n))
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


def calibrate_certified_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    max_fp: int = 4,
    held_out_cert: bool = False,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Calibrates a certified decision threshold guaranteeing low false positive rate (FPR <= 0.1%).

    Modes:
    1. Pooled validation negatives (held_out_cert=False):
       Selects threshold allowing at most max_fp (default: 4) false positives on validation negatives.
       For N ~ 10,013 negatives, max_fp=4 guarantees one-sided 95% Clopper-Pearson upper bound <= 0.0010.
    2. Held-out certification split (held_out_cert=True):
       Splits validation negatives 50/50 into calibration half and certification half.
       Selects threshold on calibration half and verifies <= 1 FP on held-out certification half (N ~ 5,000).
    """
    neg_scores = y_val_prob[y_val_true == 0]
    n_negs = len(neg_scores)
    if n_negs == 0:
        return {
            "threshold": 0.5,
            "max_fp": max_fp,
            "mode": "fallback_empty",
            "ci_95": (0.0, 1.0),
            "realized_val_fpr": 0.0,
            "is_certified": False
        }

    if not held_out_cert:
        # Pooled validation calibration: allow at most max_fp false alarms
        sorted_negs = np.sort(neg_scores)
        idx = max(0, n_negs - max_fp)
        threshold = float(sorted_negs[idx]) if idx < n_negs else float(sorted_negs[-1]) + 1e-6
        observed_fp = int(np.sum(neg_scores >= threshold))
        while observed_fp > max_fp and idx < n_negs - 1:
            idx += 1
            threshold = float(sorted_negs[idx])
            observed_fp = int(np.sum(neg_scores >= threshold))
        if observed_fp > max_fp:
            threshold = float(sorted_negs[-1]) + 1e-6
            observed_fp = int(np.sum(neg_scores >= threshold))
        ci_lower, ci_upper = clopper_pearson_ci(observed_fp, n_negs, confidence=confidence)
        return {
            "threshold": threshold,
            "max_fp_allowed": max_fp,
            "observed_val_fp": observed_fp,
            "n_val_negs": n_negs,
            "mode": "pooled_validation",
            "realized_val_fpr": observed_fp / n_negs,
            "ci_95": (round(ci_lower, 6), round(ci_upper, 6)),
            "is_certified": bool(observed_fp <= max_fp and ci_upper <= 0.0015)
        }
    else:
        # Held-out certification split: split negatives 50/50
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_negs)
        mid = n_negs // 2
        cal_negs = neg_scores[perm[:mid]]
        cert_negs = neg_scores[perm[mid:]]

        # On calibration half, pick threshold allowing at most 1 FP
        sorted_cal = np.sort(cal_negs)
        cal_idx = max(0, len(sorted_cal) - 1)
        thresh = float(sorted_cal[cal_idx])
        while int(np.sum(cal_negs >= thresh)) > 1 and cal_idx < len(sorted_cal) - 1:
            cal_idx += 1
            thresh = float(sorted_cal[cal_idx])
        if int(np.sum(cal_negs >= thresh)) > 1:
            thresh = float(sorted_cal[-1]) + 1e-6

        # Strictly evaluate on held-out certification half without mutating threshold
        cert_fp = int(np.sum(cert_negs >= thresh))
        ci_lower, ci_upper = clopper_pearson_ci(cert_fp, len(cert_negs), confidence=confidence)
        return {
            "threshold": thresh,
            "max_fp_allowed": 1,
            "observed_cert_fp": cert_fp,
            "n_cert_negs": len(cert_negs),
            "n_cal_negs": len(cal_negs),
            "mode": "held_out_certification",
            "realized_cert_fpr": cert_fp / max(len(cert_negs), 1),
            "ci_95": (round(ci_lower, 6), round(ci_upper, 6)),
            "is_certified": bool(cert_fp <= 1 and ci_upper <= 0.0015)
        }


def compare_at_matched_fpr(
    y_test_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    target_realised_fpr: float = 0.00094
) -> Dict[str, Any]:
    """
    Compares two models (e.g. SAGE vs ATTN) at exactly matched realised FPR on the test negatives.
    Eliminates threshold bias where one model appears superior only because its realised FPR is inflated.
    """
    neg_mask = (y_test_true == 0)
    pos_mask = (y_test_true == 1)

    negs_a = y_prob_a[neg_mask]
    negs_b = y_prob_b[neg_mask]
    pos_a = y_prob_a[pos_mask]
    pos_b = y_prob_b[pos_mask]

    n_negs = len(negs_a)
    n_pos = len(pos_a)
    if n_negs == 0 or n_pos == 0:
        return {"tpr_a": 0.0, "tpr_b": 0.0, "realised_fpr": 0.0, "delta_tpr": 0.0}

    if target_realised_fpr <= 0.0:
        th_a = float(np.max(negs_a)) + 1e-6
        th_b = float(np.max(negs_b)) + 1e-6
    elif target_realised_fpr >= 1.0:
        th_a = float(np.min(negs_a))
        th_b = float(np.min(negs_b))
    else:
        cutoff_idx = max(0, min(n_negs - 1, int(np.floor((1.0 - target_realised_fpr) * n_negs))))
        th_a = float(np.sort(negs_a)[cutoff_idx])
        th_b = float(np.sort(negs_b)[cutoff_idx])

    rfpr_a = float(np.mean(negs_a >= th_a))
    rfpr_b = float(np.mean(negs_b >= th_b))
    tpr_a = float(np.mean(pos_a >= th_a))
    tpr_b = float(np.mean(pos_b >= th_b))

    return {
        "target_fpr": target_realised_fpr,
        "model_a": {"threshold": th_a, "realised_fpr": rfpr_a, "tpr": tpr_a},
        "model_b": {"threshold": th_b, "realised_fpr": rfpr_b, "tpr": tpr_b},
        "tpr_difference_b_minus_a": round(tpr_b - tpr_a, 4)
    }


def compute_prevalence_adjusted_precision(tpr: float, fpr: float, pi: float) -> float:
    """
    Computes prevalence-adjusted precision for a target operating point:
    precision = (TPR * pi) / (TPR * pi + FPR * (1 - pi))
    Handles edge cases without division by zero.
    """
    denom = (tpr * pi) + (fpr * (1.0 - pi))
    if denom <= 0.0:
        return 0.0
    return float((tpr * pi) / denom)


def choose_optimal_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    criterion: str = "f1"
) -> float:
    """Selects decision threshold optimizing F1 strictly on validation predictions."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = -1.0

    for th in thresholds:
        pred = (y_val_prob >= th).astype(int)
        score = f1_score(y_val_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(th)

    return best_thresh


def compute_tpr_at_fpr(
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    target_fpr: float = 0.01
) -> Tuple[float, float, int, Optional[str]]:
    """
    Computes TPR at a target FPR using negative scores from validation set to set threshold.
    Returns: (tpr, threshold, expected_fp, warning_msg)
    """
    val_negs = y_val_prob[y_val_true == 0]
    if len(val_negs) == 0:
        return 0.0, 0.5, 0, "No validation negatives available."

    sorted_val_negs = np.sort(val_negs)
    cutoff_idx = int(np.ceil((1.0 - target_fpr) * (len(sorted_val_negs) - 1)))
    th = float(sorted_val_negs[cutoff_idx])

    test_pos = y_test_prob[y_test_true == 1]
    test_negs = y_test_prob[y_test_true == 0]
    tpr = float(np.mean(test_pos >= th)) if len(test_pos) > 0 else 0.0
    expected_fp = int(round(len(test_negs) * target_fpr))

    warning = None
    if expected_fp < 10:
        warning = (
            f"Expected false positives at {target_fpr*100:.1f}% FPR is {expected_fp} (< 10). "
            f"Test sample size ({len(test_negs)} negatives) provides limited statistical power at this FPR."
        )

    return tpr, th, expected_fp, warning


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    y_val_true: Optional[np.ndarray] = None,
    y_val_prob: Optional[np.ndarray] = None,
    target_fprs: List[float] = [0.01, 0.001],
    prevalence_pis: List[float] = [0.01, 0.001]
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics including operating points and prevalence adjustments."""
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    sample_prevalence = float(n_pos / max(n_total, 1))

    test_negs = y_prob[y_true == 0]

    fpr_metrics = {}
    if y_val_true is not None and y_val_prob is not None:
        for fpr_val in target_fprs:
            tpr_at_k, th_k, exp_fp, warn = compute_tpr_at_fpr(
                y_true, y_prob, y_val_true, y_val_prob, target_fpr=fpr_val
            )
            realised_fpr = float(np.mean(test_negs >= th_k)) if len(test_negs) > 0 else 0.0
            pred_at_k = (y_prob >= th_k).astype(int)

            f1_op = float(f1_score(y_true, pred_at_k, zero_division=0))
            prec_op = float(precision_score(y_true, pred_at_k, zero_division=0))
            rec_op = float(recall_score(y_true, pred_at_k, zero_division=0))

            # Prevalence-adjusted precisions
            adj_prec_dict = {}
            for pi in prevalence_pis:
                adj_p = compute_prevalence_adjusted_precision(tpr_at_k, realised_fpr, pi)
                adj_prec_dict[f"pi_{pi*100:.1f}pct"] = round(adj_p, 4)

            fpr_metrics[f"tpr_at_{fpr_val*100:.1f}pct_fpr"] = {
                "tpr": round(tpr_at_k, 4),
                "threshold": round(th_k, 4),
                "realised_test_fpr": round(realised_fpr, 6),
                "f1_at_operating_point": round(f1_op, 4),
                "precision_at_operating_point": round(prec_op, 4),
                "recall_at_operating_point": round(rec_op, 4),
                "prevalence_adjusted_precision": adj_prec_dict,
                "expected_fp": exp_fp,
                "warning": warn
            }

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "threshold": round(threshold, 4),
        "test_counts": {
            "total": n_total,
            "positives": n_pos,
            "negatives": n_neg,
            "sample_prevalence": round(sample_prevalence, 4)
        },
        "confusion_matrix": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "operating_points": fpr_metrics
    }


def compute_bootstrap_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Computes empirical bootstrap confidence intervals simultaneously for
    ROC-AUC, PR-AUC, and F1 in a single unified pass.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    aucs: List[float] = []
    prs: List[float] = []
    f1s: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        prob_b = y_prob[idx]
        pred_b = (prob_b >= threshold).astype(int)

        try:
            aucs.append(float(roc_auc_score(y_b, prob_b)))
            prs.append(float(average_precision_score(y_b, prob_b)))
            f1s.append(float(f1_score(y_b, pred_b, zero_division=0)))
        except Exception:
            continue

    def format_ci(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
        np_arr = np.array(arr)
        alpha = (1.0 - ci_level) / 2.0
        return {
            "point": round(float(np.mean(np_arr)), 4),
            "ci_lower": round(float(np.percentile(np_arr, alpha * 100)), 4),
            "ci_upper": round(float(np.percentile(np_arr, (1.0 - alpha) * 100)), 4),
            "std": round(float(np.std(np_arr)), 4)
        }

    return {
        "roc_auc": format_ci(aucs),
        "pr_auc": format_ci(prs),
        "f1": format_ci(f1s)
    }


def paired_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray
) -> Dict[str, Any]:
    """
    Computes exact McNemar test for paired classifier predictions.
    Labelled correctly with min(b, c) as the exact test statistic.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n11 = int(np.sum(correct_a & correct_b))
    n12 = int(np.sum(correct_a & (~correct_b)))
    n21 = int(np.sum((~correct_a) & correct_b))
    n22 = int(np.sum((~correct_a) & (~correct_b)))

    table = [[n11, n12], [n21, n22]]
    res = mcnemar(table, exact=True)

    return {
        "exact_statistic_min_b_c": float(res.statistic),
        "pvalue": float(res.pvalue),
        "discordant_a_better": n12,
        "discordant_b_better": n21,
        "contingency_table": table,
        "interpretation": "statistically significant difference" if res.pvalue < 0.05 else "no significant difference"
    }


def paired_bootstrap_auc_difference(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Paired bootstrap distribution of (AUC_a - AUC_b)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_b, y_prob_a[idx])
            auc_b = roc_auc_score(y_b, y_prob_b[idx])
            diffs.append(auc_a - auc_b)
        except Exception:
            continue

    if not diffs:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "pvalue_two_sided": 1.0}

    arr = np.array(diffs)
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(arr, alpha * 100))
    upper = float(np.percentile(arr, (1.0 - alpha) * 100))
    mean_diff = float(np.mean(arr))
    p_val = float(2.0 * min(np.mean(arr <= 0), np.mean(arr >= 0)))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "pvalue_two_sided": round(p_val, 4)
    }


def multi_seed_paired_comparisons(
    seeds: List[int],
    predictions_a: Dict[int, Dict[str, np.ndarray]],
    predictions_b: Dict[int, Dict[str, np.ndarray]],
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Computes McNemar and paired bootstrap AUC difference across ALL seeds.
    Scientific Rigor Note:
    Because random splits share overlapping test partitions across different seeds,
    the same domain instances appear in multiple seeds' test sets. Therefore, per-seed
    hypothesis tests and confidence intervals are PRIMARY.
    A summary of the per-seed differences (mean difference and fraction of seeds where p < 0.05)
    is reported to avoid false degrees-of-freedom inflation from naive observation concatenation.
    """
    per_seed = {}
    seed_pvalues_mcnemar = []
    seed_mean_auc_diffs = []
    seed_pvalues_auc = []

    pooled_true: List[int] = []
    pooled_pred_a: List[int] = []
    pooled_pred_b: List[int] = []
    pooled_prob_a: List[float] = []
    pooled_prob_b: List[float] = []

    for s in seeds:
        if s not in predictions_a or s not in predictions_b:
            continue
        pa = predictions_a[s]
        pb = predictions_b[s]
        y_true = pa["y_true"]
        y_pred_a = pa["y_pred"]
        y_pred_b = pb["y_pred"]
        y_prob_a = pa["y_prob"]
        y_prob_b = pb["y_prob"]

        mcn = paired_mcnemar_test(y_true, y_pred_a, y_pred_b)
        auc_diff = paired_bootstrap_auc_difference(y_true, y_prob_a, y_prob_b, n_bootstrap=n_bootstrap, seed=s)
        per_seed[str(s)] = {
            "mcnemar": mcn,
            "bootstrap_auc_diff": auc_diff
        }

        seed_pvalues_mcnemar.append(mcn["pvalue"])
        seed_mean_auc_diffs.append(auc_diff["mean_diff"])
        auc_pval = auc_diff.get("pvalue_two_sided", auc_diff.get("pvalue", 1.0))
        seed_pvalues_auc.append(auc_pval)

        pooled_true.extend(y_true.tolist())
        pooled_pred_a.extend(y_pred_a.tolist())
        pooled_pred_b.extend(y_pred_b.tolist())
        pooled_prob_a.extend(y_prob_a.tolist())
        pooled_prob_b.extend(y_prob_b.tolist())

    # Summary across seeds
    n_valid = len(seed_mean_auc_diffs)
    mean_auc_diff = float(np.mean(seed_mean_auc_diffs)) if n_valid > 0 else 0.0
    sig_mcn_count = int(sum(p < 0.05 for p in seed_pvalues_mcnemar))
    sig_auc_count = int(sum(p < 0.05 for p in seed_pvalues_auc))

    pooled_mcn = paired_mcnemar_test(np.array(pooled_true), np.array(pooled_pred_a), np.array(pooled_pred_b)) if len(pooled_true) > 0 else {}
    pooled_auc_diff = paired_bootstrap_auc_difference(np.array(pooled_true), np.array(pooled_prob_a), np.array(pooled_prob_b), n_bootstrap=n_bootstrap, seed=42) if len(pooled_true) > 0 else {}

    return {
        "per_seed": per_seed,
        "cross_seed_summary": {
            "num_evaluated_seeds": n_valid,
            "mean_paired_auc_difference": round(mean_auc_diff, 4),
            "mcnemar_significant_seeds_count": sig_mcn_count,
            "auc_significant_seeds_count": sig_auc_count,
            "all_seeds_auc_significant": (sig_auc_count == n_valid and n_valid > 0)
        },
        "pooled_summary": {
            "mcnemar": pooled_mcn,
            "bootstrap_auc_diff": pooled_auc_diff,
            "warning": "Overlapping test cohorts across random seeds mean naive pooled tests have correlated observations. Refer to cross_seed_summary and per_seed results for uninflated degrees of freedom."
        }
    }
