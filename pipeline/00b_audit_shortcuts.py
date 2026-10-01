#!/usr/bin/env python3
"""
CLI script to audit single-feature shortcuts and class-source confounding.
Runs after parsing and writes audit_shortcuts.json and audit_shortcuts.md.
"""

import argparse
import sys
from pathlib import Path

# Set stdout/stderr to utf-8 if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.data.audit import run_shortcut_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset shortcuts and class confounding.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    parquet_file = paths.processed_dir / "domains.parquet"
    out_dir = paths.results_dir / "profile"
    paths.check_write_path(out_dir)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Running shortcut audit on: {parquet_file}...")
    summary, md_text = run_shortcut_audit(parquet_file, out_dir, is_fixture=args.fixture)

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"\n{pfx}=== SHORTCUT AUDIT COMPLETE ===")
    print(f"Report JSON: {out_dir / 'audit_shortcuts.json'}")
    print(f"Report Markdown: {out_dir / 'audit_shortcuts.md'}")
    print("Single feature AUCs:")
    for k, v in summary["single_feature_aucs"].items():
        print(f"  - {k}: {v:.4f}")
    if summary["features_with_auc_ge_0_90"]:
        print(f"[WARNING] Features with AUC >= 0.90: {summary['features_with_auc_ge_0_90']}")
    else:
        print("[PASS] Verdict: No single domain feature has AUC >= 0.90.")


if __name__ == "__main__":
    main()
