#!/usr/bin/env python3
"""
CLI script to profile Zenodo DomainRadar v2 dataset files.
Streams raw JSON files, generates profile_report.json, profile_report.md (<=200 lines),
and time_split_valid.json.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.data.profiler import profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Zenodo dataset files.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/zenodo")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    out_profile_dir = paths.results_dir / "profile"
    paths.check_write_path(out_profile_dir)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Starting dataset profiling...")
    print(f"  Input raw dir: {paths.raw_dir}")
    print(f"  Output dir: {out_profile_dir}")

    min_months = cfg.get("profiling", {}).get("min_distinct_months_for_time_split", 4)
    report, md_text, time_split = profile_dataset(
        raw_dir=paths.raw_dir,
        output_dir=out_profile_dir,
        min_distinct_months=min_months,
        is_fixture=args.fixture
    )

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"\n{pfx}=== PROFILING COMPLETE ===")
    print(f"Report JSON: {out_profile_dir / 'profile_report.json'}")
    print(f"Report Markdown: {out_profile_dir / 'profile_report.md'}")
    print(f"Time Split Valid: {time_split['time_split_valid']} (Overlapping months: {time_split['overlapping_months_count']})")
    print(f"Markdown line count: {len(md_text.splitlines())} lines (must be <= 200)")


if __name__ == "__main__":
    main()
