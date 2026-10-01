"""
Shortcut and leakage audit for parsed domain records.
Evaluates single-feature ROC-AUCs to detect class confounding, dataset artifacts,
and source proxies.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_binary_feature_auc(y_true: np.ndarray, feature_vals: np.ndarray) -> float:
    """Computes AUC for a binary or scalar feature, taking max(AUC, 1 - AUC) if inversely correlated."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        auc = roc_auc_score(y_true, feature_vals)
        return float(max(auc, 1.0 - auc))
    except Exception:
        return 0.5


def run_shortcut_audit(
    parquet_path: Path,
    output_dir: Path,
    is_fixture: bool = False
) -> Tuple[Dict[str, Any], str]:
    """
    Computes single-feature AUCs, source-by-class distribution, and produces
    audit_shortcuts.json and audit_shortcuts.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    y = df["label"].to_numpy().astype(int)

    # 1. Feature extractions
    # Length
    lens = df["domain"].str.len().to_numpy()
    auc_len = compute_binary_feature_auc(y, lens)

    # Num labels
    num_labels = df["domain"].str.count(r"\.").to_numpy() + 1
    auc_num_labels = compute_binary_feature_auc(y, num_labels)

    # Subdomain flag
    sub_flags = df["has_subdomain"].astype(int).to_numpy()
    auc_subdomain = compute_binary_feature_auc(y, sub_flags)

    # TLD frequency target encoding
    tlds = df["domain"].apply(lambda x: "." + x.split(".")[-1] if "." in x else "").to_numpy()
    tld_counts = Counter(tlds)
    tld_mal_rates = {}
    for tld_val in tld_counts.keys():
        mask = (tlds == tld_val)
        tld_mal_rates[tld_val] = float(np.mean(y[mask]))
    tld_scores = np.array([tld_mal_rates[t] for t in tlds])
    auc_tld = compute_binary_feature_auc(y, tld_scores)

    # NXDOMAIN / No-IP
    no_ip_flags = df["no_ip"].astype(int).to_numpy()
    auc_no_ip = compute_binary_feature_auc(y, no_ip_flags)

    # Has TLS
    has_tls_flags = (~df["no_tls"]).astype(int).to_numpy()
    auc_has_tls = compute_binary_feature_auc(y, has_tls_flags)

    # Has RDAP
    has_rdap_flags = (~df["no_rdap"]).astype(int).to_numpy()
    auc_has_rdap = compute_binary_feature_auc(y, has_rdap_flags)

    # Source-only classifier
    sources = df["source"].to_numpy()
    src_mal_rates = {}
    for s_val in np.unique(sources):
        mask = (sources == s_val)
        src_mal_rates[s_val] = float(np.mean(y[mask]))
    src_scores = np.array([src_mal_rates[s] for s in sources])
    auc_source = compute_binary_feature_auc(y, src_scores)

    auc_results = {
        "length": round(auc_len, 4),
        "num_labels": round(auc_num_labels, 4),
        "subdomain_flag": round(auc_subdomain, 4),
        "tld": round(auc_tld, 4),
        "no_ip_nxdomain": round(auc_no_ip, 4),
        "has_tls": round(auc_has_tls, 4),
        "has_rdap": round(auc_has_rdap, 4),
        "source_only": round(auc_source, 4)
    }

    # Class by source table
    source_table: Dict[str, Dict[str, int]] = {}
    for s_val in np.unique(sources):
        mask = (sources == s_val)
        source_table[str(s_val)] = {
            "malware_count": int(np.sum(y[mask] == 1)),
            "benign_count": int(np.sum(y[mask] == 0))
        }

    # Verdict on domain/infra features (excluding data-provenance source tag)
    domain_feature_aucs = {k: v for k, v in auc_results.items() if k != "source_only"}
    shortcuts_found = [k for k, v in domain_feature_aucs.items() if v >= 0.90]
    verdict_passed = len(shortcuts_found) == 0

    audit_summary = {
        "is_fixture": is_fixture,
        "single_feature_aucs": auc_results,
        "features_with_auc_ge_0_90": shortcuts_found,
        "verdict_passed": verdict_passed,
        "source_by_class_table": source_table
    }

    json_path = output_dir / "audit_shortcuts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Markdown report
    pfx = "[FIXTURE - meaningless] " if is_fixture else ""
    md_lines: List[str] = [
        f"# {pfx}Dataset Shortcut & Leakage Audit",
        "",
        "## 1. Single-Feature ROC-AUCs",
        "*(Domain and infrastructure features with AUC >= 0.90 indicate potential trivial shortcuts)*",
        "",
        "| Feature | ROC-AUC | Status |",
        "|---|---|---|"
    ]
    for feat, score in auc_results.items():
        if feat == "source_only":
            status = "Provenance Metadata (Excluded from Features)"
        elif score >= 0.90:
            status = "WARNING: SHORTCUT (>= 0.90)"
        else:
            status = "PASS: OK (< 0.90)"
        md_lines.append(f"| `{feat}` | {score:.4f} | {status} |")

    md_lines.append("")
    md_lines.append("## 2. Class-by-Source Contingency Table")
    md_lines.append("| Source Feed | Malware (1) | Benign (0) | Malware Ratio |")
    md_lines.append("|---|---|---|---|")
    for s_name, counts in sorted(source_table.items()):
        total = counts["malware_count"] + counts["benign_count"]
        ratio = counts["malware_count"] / max(total, 1)
        md_lines.append(f"| {s_name} | {counts['malware_count']:,} | {counts['benign_count']:,} | {ratio*100:.1f}% |")

    md_lines.append("")
    md_lines.append("## 3. Audit Verdict")
    if verdict_passed:
        md_lines.append("> **VERDICT: PASS.** No single domain or infrastructure feature trivially separates malware from benign (all domain AUCs < 0.90).")
    else:
        md_lines.append(f"> **VERDICT: WARNING.** Identified shortcut features with AUC >= 0.90: {', '.join(shortcuts_found)}.")
        if "subdomain_flag" in shortcuts_found:
            md_lines.append("> Subdomain flag is confirmed to be a shortcut proxy; keep `include_has_subdomain: false` in config.")

    md_text = "\n".join(md_lines) + "\n"
    md_path = output_dir / "audit_shortcuts.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    return audit_summary, md_text
