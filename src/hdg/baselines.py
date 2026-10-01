"""
Baseline models evaluated on identical splits for Malicious-HDG:
1. Lexical: Char n-gram TF-IDF (2-5) + Logistic Regression on e2LD.
2. Tabular: XGBoost on tabular non-lexical features.
3. Tabular + Lexical: XGBoost on tabular + lexical features.
4. Single-Feature: Domain length and No-IP baselines.
5. Isolated Domain MLP: HeteroGNN with use_edges=False (isolates graph marginal value).
All hyperparameters and thresholds tuned strictly on validation splits.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import torch
import torch.nn as nn

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.graph import extract_domain_feature_vector, extract_domain_features_vectorized
from src.hdg.metrics import (
    compute_bootstrap_cis,
    choose_optimal_threshold,
    evaluate_predictions,
)
from src.hdg.splits import DataSplits
from src.hdg.train import train_eval_gnn


def run_tfidf_lexical_baseline(
    df: pd.DataFrame,
    splits: DataSplits,
    cfg: Dict[str, Any],
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Char n-gram TF-IDF (2-5) + Logistic Regression on e2LD."""
    b_cfg = cfg.get("baselines", {}).get("tfidf", {})
    ngram_range = tuple(b_cfg.get("ngram_range", [2, 5]))
    min_df = int(b_cfg.get("min_df", 5))
    max_features = int(b_cfg.get("max_features", 10000))
    c_val = float(b_cfg.get("C", 1.0))

    train_df = df.iloc[splits.train_indices]
    val_df = df.iloc[splits.val_indices]
    test_df = df.iloc[splits.test_indices]

    x_train_text = train_df["e2LD"].tolist()
    x_val_text = val_df["e2LD"].tolist()
    x_test_text = test_df["e2LD"].tolist()

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
    )
    x_train_vec = vec.fit_transform(x_train_text)
    x_val_vec = vec.transform(x_val_text)
    x_test_vec = vec.transform(x_test_text)

    clf = LogisticRegression(C=c_val, max_iter=500, random_state=seed)
    clf.fit(x_train_vec, y_train)

    val_probs = clf.predict_proba(x_val_vec)[:, 1]
    test_probs = clf.predict_proba(x_test_vec)[:, 1]

    opt_th = choose_optimal_threshold(y_val, val_probs)
    test_eval = evaluate_predictions(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        y_val_true=y_val,
        y_val_prob=val_probs
    )

    test_eval["bootstrap_ci"] = compute_bootstrap_cis(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        seed=seed
    )

    return test_eval, y_test, test_probs


def run_xgboost_baseline(
    df: pd.DataFrame,
    splits: DataSplits,
    cfg: Dict[str, Any],
    use_lexical: bool = False,
    drop_tld: bool = False,
    drop_null_rates: bool = False,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """XGBoost classifier on tabular features (with or without lexical stats and feature-group ablations)."""
    xgb_cfg = cfg.get("baselines", {}).get("xgboost", {})
    n_estimators = int(xgb_cfg.get("n_estimators", 100))
    max_depth = int(xgb_cfg.get("max_depth", 6))
    lr = float(xgb_cfg.get("learning_rate", 0.1))

    p_cfg = cfg.get("parsing", {})
    include_sub = p_cfg.get("include_has_subdomain", False)

    X_all = extract_domain_features_vectorized(
        df,
        use_lexical=use_lexical,
        include_has_subdomain=include_sub
    )

    # Optional feature group ablation
    if drop_null_rates:
        # Zero out no_ip, no_rdap, no_tls (indices 0, 1, 2)
        X_all[:, :3] = 0.0

    y_all = df["label"].to_numpy().astype(int)

    X_train, y_train = X_all[splits.train_indices], y_all[splits.train_indices]
    X_val, y_val = X_all[splits.val_indices], y_all[splits.val_indices]
    X_test, y_test = X_all[splits.test_indices], y_all[splits.test_indices]

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    opt_th = choose_optimal_threshold(y_val, val_probs)
    test_eval = evaluate_predictions(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        y_val_true=y_val,
        y_val_prob=val_probs
    )

    test_eval["bootstrap_ci"] = compute_bootstrap_cis(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        seed=seed
    )

    return test_eval, y_test, test_probs


def run_single_feature_baseline(
    df: pd.DataFrame,
    splits: DataSplits,
    feature_name: str = "length",
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Single-feature baseline classifier (e.g. domain length or no_ip)."""
    if feature_name == "length":
        vals = np.array([len(str(d)) for d in df["domain"]], dtype=np.float32).reshape(-1, 1)
    elif feature_name == "no_ip":
        vals = df["no_ip"].fillna(False).astype(np.float32).to_numpy().reshape(-1, 1)
    elif feature_name in df.columns:
        vals = df[feature_name].fillna(0).astype(np.float32).to_numpy().reshape(-1, 1)
    else:
        raise ValueError(f"Unknown feature for single-feature baseline: {feature_name}")

    y_all = df["label"].to_numpy().astype(int)

    X_train, y_train = vals[splits.train_indices], y_all[splits.train_indices]
    X_val, y_val = vals[splits.val_indices], y_all[splits.val_indices]
    X_test, y_test = vals[splits.test_indices], y_all[splits.test_indices]

    clf = LogisticRegression(random_state=seed)
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    opt_th = choose_optimal_threshold(y_val, val_probs)
    test_eval = evaluate_predictions(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        y_val_true=y_val,
        y_val_prob=val_probs
    )

    test_eval["bootstrap_ci"] = compute_bootstrap_cis(
        y_true=y_test,
        y_prob=test_probs,
        threshold=opt_th,
        seed=seed
    )

    return test_eval, y_test, test_probs


def run_mlp_no_edges_baseline(
    scaled_data: Any,
    splits: DataSplits,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    epochs: int = 30,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Isolated Domain MLP baseline (HeteroGNN with use_edges=False)."""
    in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
    model = HeteroGNN(
        metadata=scaled_data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=64,
        out_dim=32,
        num_layers=2,
        use_edges=False
    )

    test_eval, y_test, test_probs, _ = train_eval_gnn(
        model=model,
        data=scaled_data,
        splits=splits,
        df=df,
        epochs=epochs,
        seed=seed
    )

    return test_eval, y_test, test_probs
