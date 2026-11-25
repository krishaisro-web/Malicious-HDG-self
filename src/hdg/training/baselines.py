"""
Baseline models evaluated on identical splits for Malicious-HDG:
1. Lexical: Char n-gram TF-IDF (2-5) + Logistic Regression on e2LD.
2. Tabular: XGBoost on tabular non-lexical features.
3. Tabular + Lexical: XGBoost on tabular + lexical features.
4. Single-Feature: Domain length and No-IP baselines.
5. Isolated Domain MLP: HeteroGNN with use_edges=False (isolates graph marginal value).
All hyperparameters and thresholds tuned strictly on validation splits.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.data.graph import extract_domain_features_vectorized
from src.hdg.data.features import (
    _CLOUD_CDN_KW,
    _FREE_AUTO_CA_KW,
    _HOSTING_VPS_KW,
    _MAJOR_DNS_CDN_KW,
    _PAID_EV_CA_KW,
    get_feature_indices,
)
from src.hdg.metrics import (
    compute_bootstrap_cis,
    choose_optimal_threshold,
    evaluate_predictions,
)
from src.hdg.data.splits import DataSplits
from src.hdg.training.train import train_eval_gnn


def extract_1hop_infra_features_for_tabular(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts 10 flattened 1-hop infrastructure summary features per domain (zero label leakage)
    for fair tabular XGBoost comparison (RFC-003 Section 2.4, Objective O3).
    """
    n = len(df)
    if n == 0:
        return np.empty((0, 10), dtype=np.float32)

    mat = np.zeros((n, 10), dtype=np.float32)
    col_asns = df["asns"].tolist() if "asns" in df.columns else [None] * n
    col_ip_recs = df["ip_records"].tolist() if "ip_records" in df.columns else [None] * n
    col_ns = df["nameservers"].tolist() if "nameservers" in df.columns else [None] * n
    col_cert_org = df["cert_org"].tolist() if "cert_org" in df.columns else [None] * n
    col_cert_self = df["cert_is_self_signed"].tolist() if "cert_is_self_signed" in df.columns else [False] * n
    col_reg_w = df["reg_has_whois"].tolist() if "reg_has_whois" in df.columns else [False] * n
    col_reg_e = df["reg_has_email"].tolist() if "reg_has_email" in df.columns else [False] * n
    col_reg_u = df["reg_has_url"].tolist() if "reg_has_url" in df.columns else [False] * n
    col_reg_t = df["reg_has_tel"].tolist() if "reg_has_tel" in df.columns else [False] * n

    for i in range(n):
        asns_i = col_asns[i]
        asn_cnt = float(len(asns_i)) if isinstance(asns_i, (list, np.ndarray)) else 0.0

        ip_recs = col_ip_recs[i]
        cloud_hits = 0
        host_hits = 0
        geo_miss = 0
        plen_sum = 0.0
        plen_cnt = 0
        rec_cnt = 0
        if isinstance(ip_recs, (list, np.ndarray)):
            for r in ip_recs:
                if not isinstance(r, dict):
                    continue
                rec_cnt += 1
                org_s = str(r.get("asn_org") or "").lower()
                if any(k in org_s for k in _CLOUD_CDN_KW):
                    cloud_hits += 1
                if any(k in org_s for k in _HOSTING_VPS_KW):
                    host_hits += 1
                if not (r.get("country_code") or r.get("country")):
                    geo_miss += 1
                if r.get("prefix_len") is not None:
                    plen_sum += float(r["prefix_len"]) / 32.0
                    plen_cnt += 1

        ns_list = col_ns[i]
        ns_major_share = 0.0
        if isinstance(ns_list, (list, np.ndarray)) and len(ns_list) > 0:
            maj = sum(1 for ns in ns_list if any(k in str(ns).lower() for k in _MAJOR_DNS_CDN_KW))
            ns_major_share = float(maj / len(ns_list))

        c_org = str(col_cert_org[i] or "").lower()
        c_free = 1.0 if any(k in c_org for k in _FREE_AUTO_CA_KW) else 0.0
        c_paid = 1.0 if any(k in c_org for k in _PAID_EV_CA_KW) else 0.0
        c_self = 1.0 if bool(col_cert_self[i]) else 0.0
        reg_comp = float(
            int(bool(col_reg_w[i])) + int(bool(col_reg_e[i])) + int(bool(col_reg_u[i])) + int(bool(col_reg_t[i]))
        ) / 4.0

        mat[i] = [
            np.log1p(asn_cnt),
            float(cloud_hits / max(1, rec_cnt)),
            float(host_hits / max(1, rec_cnt)),
            float(geo_miss / max(1, rec_cnt)) if rec_cnt > 0 else 1.0,
            float(plen_sum / max(1, plen_cnt)),
            c_free,
            c_paid,
            c_self,
            reg_comp,
            ns_major_share,
        ]
    return mat


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

    train_col = "e2LD" if "e2LD" in train_df.columns else "domain"
    val_col = "e2LD" if "e2LD" in val_df.columns else "domain"
    test_col = "e2LD" if "e2LD" in test_df.columns else "domain"

    x_train_text = train_df[train_col].astype(str).tolist()
    x_val_text = val_df[val_col].astype(str).tolist()
    x_test_text = test_df[test_col].astype(str).tolist()

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    effective_min_df = min(min_df, max(1, len(x_train_text) // 2)) if len(x_train_text) < 10 else min_df
    vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        min_df=effective_min_df,
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
    include_1hop_infra: Optional[bool] = None,
    seed: int = 42
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """XGBoost classifier on tabular features (with optional lexical stats and fair 1-hop infra attrs)."""
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

    # Optional feature group ablation resolved dynamically via registry
    if drop_null_rates:
        miss_indices = get_feature_indices(
            "domain", group="missingness", include_has_subdomain=include_sub, dim=X_all.shape[1]
        )
        if miss_indices:
            X_all[:, miss_indices] = 0.0

    # Fair 1-hop flattened infrastructure alignment for xgboost_tabular_lexical (RFC-003 Section 2.4)
    if include_1hop_infra is None:
        include_1hop_infra = use_lexical
    if include_1hop_infra:
        X_infra_1hop = extract_1hop_infra_features_for_tabular(df)
        X_all = np.hstack([X_all, X_infra_1hop]).astype(np.float32)

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
    m_cfg = cfg.get("model", {})
    has_cb = hasattr(scaled_data["domain"], "char_bytes") and scaled_data["domain"].char_bytes is not None
    model = HeteroGNN(
        metadata=scaled_data.metadata(),
        in_channels_dict=in_channels,
        hidden_dim=64,
        out_dim=32,
        num_layers=2,
        use_edges=False,
        use_char_cnn=bool(m_cfg.get("use_char_cnn", False) and has_cb),
        char_cnn_dim=int(m_cfg.get("char_cnn_dim", 16)),
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
