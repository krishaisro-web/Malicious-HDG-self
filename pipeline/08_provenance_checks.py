#!/usr/bin/env python3
"""
Shortcut, provenance, and collection artifact audit for Malicious-HDG:
1. Benign-vs-benign classifier (Cisco Umbrella vs CESNET)
2. Cross-source generalization: Train on {primary malware source + Umbrella}, test on {other malware sources + CESNET}
3. Feature-group sensitivity ablation (excluding null-rate & missingness features)
4. Graph connectivity audit: Share of domains per class connected to reused infrastructure (degree >= 2)
Outputs provenance_checks.json and provenance_checks.md with plain-language verdicts.
"""

from typing import Optional
import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.config import load_config, get_resolved_paths, init_thread_pool
from src.hdg.data.graph import extract_domain_features_vectorized, apply_split_scaling
from src.hdg.data.splits import DataSplits
from src.hdg.training.train import train_eval_gnn


def run_benign_vs_benign(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Classifies Cisco Umbrella vs CESNET benign domains to measure source divergence."""
    benign_df = df[df["label"] == 0].copy().reset_index(drop=True)
    if len(benign_df) == 0 or len(benign_df["source"].unique()) < 2:
        return {"status": "Insufficient multiple benign sources"}

    # Assign label 1 to Umbrella, 0 to CESNET
    benign_df["b_target"] = (benign_df["source"].str.lower().str.contains("umbrella")).astype(int)
    y = benign_df["b_target"].to_numpy()

    if len(np.unique(y)) < 2:
        return {"status": "Only one benign source present"}

    X = extract_domain_features_vectorized(benign_df, use_lexical=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)

    clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    preds = (probs >= 0.5).astype(int)
    f1 = float(f1_score(y_test, preds, zero_division=0))

    return {
        "roc_auc": round(auc, 4),
        "f1": round(f1, 4),
        "test_samples": len(y_test),
        "umbrella_ratio": round(float(np.mean(y)), 4),
        "interpretation": "High source divergence (AUC > 0.80)" if auc > 0.80 else "Low/moderate source divergence (AUC <= 0.80)"
    }


def run_cross_source_eval(
    df: pd.DataFrame,
    base_graph: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates cross-source transfer:
    Train on: {Primary malware source + Cisco Umbrella}
    Test on:  {Other malware sources + CESNET}
    """
    malware_df = df[df["label"] == 1]
    if len(malware_df) == 0:
        return {"status": "No malware records available"}

    source_counts = malware_df["source"].value_counts()
    primary_source = source_counts.index[0]

    # Split assignment
    train_mask = (
        ((df["label"] == 1) & (df["source"] == primary_source)) |
        ((df["label"] == 0) & (df["source"].str.lower().str.contains("umbrella")))
    ).to_numpy()

    test_mask = (
        ((df["label"] == 1) & (df["source"] != primary_source)) |
        ((df["label"] == 0) & (df["source"].str.lower().str.contains("cesnet")))
    ).to_numpy()

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) < 50 or len(test_idx) < 50:
        return {"status": "Insufficient cross-source partition sizes"}

    # XGBoost tabular evaluation
    X = extract_domain_features_vectorized(df, use_lexical=True)
    y = df["label"].to_numpy().astype(int)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"status": "Single-class partition in cross-source split"}

    clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    xgb_auc = float(roc_auc_score(y_test, probs))
    xgb_f1 = float(f1_score(y_test, (probs >= 0.5).astype(int), zero_division=0))

    gnn_res = None
    if base_graph is not None and cfg is not None:
        try:
            # 85/15 val split within train
            val_size = int(len(train_idx) * 0.15)
            rng = np.random.default_rng(seed)
            shuff = rng.permutation(train_idx)
            tr_idx, val_idx = shuff[val_size:], shuff[:val_size]

            splits = DataSplits(train_indices=tr_idx, val_indices=val_idx, test_indices=test_idx, split_type="cross_source")
            scaled_g, _ = apply_split_scaling(base_graph, tr_idx)
            in_ch = {nt: scaled_g[nt].x.shape[1] for nt in scaled_g.node_types}
            model = HeteroGNN(metadata=scaled_g.metadata(), in_channels_dict=in_ch, variant="sage")
            res_gnn, _, _, _ = train_eval_gnn(model, scaled_g, splits, df, epochs=10, seed=seed)
            gnn_res = {"roc_auc": res_gnn["roc_auc"], "f1": res_gnn["f1"]}
        except Exception as e:
            gnn_res = {"error": str(e)}

    return {
        "primary_malware_source": str(primary_source),
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "xgboost_cross_source": {"roc_auc": round(xgb_auc, 4), "f1": round(xgb_f1, 4)},
        "gnn_cross_source": gnn_res
    }


def run_feature_group_ablation(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Measures performance sensitivity when dropping missingness/null-rate features."""
    X_full = extract_domain_features_vectorized(df, use_lexical=True)
    # Drop null rate flags (indices 0, 1, 2)
    X_dropped = X_full.copy()
    X_dropped[:, :3] = 0.0

    y = df["label"].to_numpy().astype(int)
    X_tr_f, X_te_f, y_tr, y_te = train_test_split(X_full, y, test_size=0.20, random_state=seed, stratify=y)
    X_tr_d, X_te_d, _, _ = train_test_split(X_dropped, y, test_size=0.20, random_state=seed, stratify=y)

    clf_full = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=1)
    clf_full.fit(X_tr_f, y_tr)
    auc_full = float(roc_auc_score(y_te, clf_full.predict_proba(X_te_f)[:, 1]))

    clf_drop = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=seed, n_jobs=1)
    clf_drop.fit(X_tr_d, y_tr)
    auc_drop = float(roc_auc_score(y_te, clf_drop.predict_proba(X_te_d)[:, 1]))

    return {
        "full_auc": round(auc_full, 4),
        "no_null_rate_auc": round(auc_drop, 4),
        "delta": round(auc_drop - auc_full, 4),
        "interpretation": "Vulnerable to collection artifacts" if abs(auc_drop - auc_full) > 0.05 else "Robust to missingness features"
    }


def run_connectivity_audit(graph: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes per relation and per class the share of domains having at least
    one neighbor of degree >= 2 (explains whether the graph carries reuse signal).
    """
    labels = df["label"].to_numpy()
    n_domains = len(df)
    results = {}

    relations = [
        ("resolves_to", ("domain", "resolves_to", "ip"), "ip"),
        ("uses_ns", ("domain", "uses_ns", "nameserver"), "nameserver"),
        ("registered_by", ("domain", "registered_by", "registrar"), "registrar"),
        ("uses_cert", ("domain", "uses_cert", "certificate"), "certificate")
    ]

    for rel_name, edge_type, dst_type in relations:
        if edge_type not in graph.edge_types:
            continue
        edge_index = graph[edge_type].edge_index.cpu().numpy()
        if edge_index.shape[1] == 0:
            continue

        d_idx = edge_index[0]
        dst_idx = edge_index[1]

        # Destination degrees
        dst_counts = Counter(dst_idx)
        shared_dsts = {k for k, v in dst_counts.items() if v >= 2}

        # Domains with at least one neighbor of degree >= 2
        shared_domain_set = set(d_idx[np.isin(dst_idx, list(shared_dsts))])

        mal_domains = np.where(labels == 1)[0]
        ben_domains = np.where(labels == 0)[0]

        mal_share = float(np.mean([d in shared_domain_set for d in mal_domains])) if len(mal_domains) > 0 else 0.0
        ben_share = float(np.mean([d in shared_domain_set for d in ben_domains])) if len(ben_domains) > 0 else 0.0
        overall_share = float(len(shared_domain_set) / max(n_domains, 1))

        results[rel_name] = {
            "overall_share_degree_ge_2": round(overall_share, 4),
            "malware_share_degree_ge_2": round(mal_share, 4),
            "benign_share_degree_ge_2": round(ben_share, 4),
            "total_shared_infra_nodes": len(shared_dsts)
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provenance, shortcut, and graph connectivity checks.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached runs")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "profile"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Domains file not found: {parquet_file}")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading domain data and graph for provenance audits...")
    df = pd.read_parquet(parquet_file)
    max_d = args.max_domains or (5000 if args.smoke else None)
    if max_d is not None and len(df) > max_d:
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    base_graph = torch.load(graph_file, weights_only=False) if graph_file.exists() else None

    # 1. Benign vs Benign
    print(f"{pfx}1. Evaluating benign-vs-benign source classifier (Umbrella vs CESNET)...")
    b_vs_b = run_benign_vs_benign(df)

    # 2. Cross-Source Generalization
    print(f"{pfx}2. Evaluating cross-source generalization...")
    cross_src = run_cross_source_eval(df, base_graph=base_graph, cfg=cfg)

    # 3. Feature Group Sensitivity
    print(f"{pfx}3. Evaluating feature-group sensitivity (dropping null-rate features)...")
    feat_ablation = run_feature_group_ablation(df)

    # 4. Graph Connectivity Audit
    conn_audit = {}
    if base_graph is not None:
        print(f"{pfx}4. Performing graph infrastructure connectivity & reuse audit...")
        conn_audit = run_connectivity_audit(base_graph, df)

    report = {
        "is_fixture": args.fixture,
        "benign_vs_benign": b_vs_b,
        "cross_source_transfer": cross_src,
        "feature_group_ablation": feat_ablation,
        "graph_connectivity_audit": conn_audit
    }

    # Save JSON
    json_path = out_dir / "provenance_checks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown Report
    pfx_t = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx_t}Provenance, Collection Shortcut, and Infrastructure Connectivity Audit",
        "",
        "## 1. Benign Source Disparity (Umbrella vs CESNET)",
        f"- **ROC-AUC**: {b_vs_b.get('roc_auc', 'N/A')}",
        f"- **Interpretation**: {b_vs_b.get('interpretation', 'N/A')}",
        "",
        "## 2. Cross-Source Generalization Transfer",
        f"- **Primary Malware Feed**: {cross_src.get('primary_malware_source', 'N/A')}",
        f"- **XGBoost Transfer ROC-AUC**: {cross_src.get('xgboost_cross_source', {}).get('roc_auc', 'N/A')}",
        "",
        "## 3. Feature Sensitivity (Missingness / Null-rate Removal)",
        f"- **Full Model AUC**: {feat_ablation.get('full_auc', 'N/A')}",
        f"- **No-Null-Rate AUC**: {feat_ablation.get('no_null_rate_auc', 'N/A')}",
        f"- **AUC Delta**: {feat_ablation.get('delta', 'N/A')} ({feat_ablation.get('interpretation', 'N/A')})",
        "",
        "## 4. Graph Infrastructure Reuse Connectivity Audit",
        "| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |",
        "|---|---|---|---|---|"
    ]
    for rel_k, c_stat in conn_audit.items():
        md_lines.append(
            f"| `{rel_k}` | {c_stat['overall_share_degree_ge_2']*100:.1f}% | {c_stat['malware_share_degree_ge_2']*100:.1f}% | {c_stat['benign_share_degree_ge_2']*100:.1f}% | {c_stat['total_shared_infra_nodes']} |"
        )
    md_lines.append("")

    md_path = out_dir / "provenance_checks.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== PROVENANCE CHECKS COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
