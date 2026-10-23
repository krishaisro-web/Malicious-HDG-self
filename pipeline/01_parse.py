#!/usr/bin/env python3
"""
CLI script to stream and normalize Zenodo DomainRadar v2 dataset records.
Generates data_processed/domains.parquet, data_processed/domains_timematched.parquet,
and parsing_summary.json.
Supports --limit N for quick smoke / real-data checks.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.data.parse import parse_and_process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and normalize Zenodo DomainRadar v2 records.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--thin-client", action="store_true", help="Run with HP Thin Client profile")
    parser.add_argument("--force", action="store_true", help="Force re-parsing of raw dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records per file for quick checks")
    args = parser.parse_args()

    # Safeguard: if --limit is supplied without --fixture, route to --smoke to protect production data
    is_smoke = args.smoke
    if args.limit is not None and not args.fixture and not args.smoke:
        print("[SAFETY] --limit supplied without --fixture or --smoke flag. Automatically routing to smoke profile (data/processed/smoke) to prevent overwriting production data.")
        is_smoke = True

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=is_smoke, root_dir=REPO_ROOT)

    paths.check_write_path(paths.processed_dir)

    mode_str = "FIXTURE" if args.fixture else ("SMOKE" if is_smoke else "REAL")
    domains_file = paths.processed_dir / "domains.parquet"
    timematched_file = paths.processed_dir / "domains_timematched.parquet"
    summary_file = paths.processed_dir / "parsing_summary.json"

    # Strict cache check: verify raw inputs and parsing configuration
    can_skip = False
    if not args.force and domains_file.exists() and timematched_file.exists() and summary_file.exists():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
            # Verify configuration parameters match
            prev_e2ld = prev_summary.get("normalize_to_e2LD")
            curr_e2ld = cfg.get("parsing", {}).get("normalize_to_e2LD", True)
            prev_ratio = prev_summary.get("benign_ratio")
            curr_ratio = cfg.get("parsing", {}).get("benign_ratio", 2.0)
            if prev_e2ld == curr_e2ld and prev_ratio == curr_ratio:
                can_skip = True
        except Exception:
            can_skip = False

    if can_skip:
        print(f"[{mode_str}] Processed dataset already exists and config matched in {paths.processed_dir}. Skipping (use --force to re-parse).")
        return

    print(f"[{mode_str}] Starting dataset parsing pipeline...")
    print(f"  Raw data: {paths.raw_dir}")
    print(f"  Processed destination: {paths.processed_dir}")
    if args.limit:
        print(f"  Limit per file: {args.limit}")

    summary = parse_and_process_dataset(
        raw_dir=paths.raw_dir,
        output_dir=paths.processed_dir,
        config=cfg,
        is_fixture=args.fixture,
        limit=args.limit
    )

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"\n{pfx}=== PARSING COMPLETE ===")
    print(f"Total domains kept: {summary['final_counts']['total_primary']} (Malware: {summary['final_counts']['malware']}, Benign: {summary['final_counts']['benign_uniform']})")
    print(f"Time-matched cohort: {summary['final_counts']['total_timematched']}")
    print(f"Cross-class collisions dropped: {summary['cross_class_dropped_count']}")


if __name__ == "__main__":
    main()
