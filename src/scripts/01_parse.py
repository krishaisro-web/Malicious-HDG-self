#!/usr/bin/env python3
"""
CLI script to stream and normalize Zenodo DomainRadar v2 dataset records.
Generates data_processed/domains.parquet and parsing_summary.json.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.config import load_config, get_resolved_paths
from src.hdg.parse import parse_and_process_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and normalize Zenodo DomainRadar v2 records.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, root_dir=REPO_ROOT)

    paths.check_write_path(paths.processed_dir)

    print(f"[{'FIXTURE' if args.fixture else 'REAL'}] Starting dataset parsing pipeline...")
    print(f"  Raw data: {paths.raw_dir}")
    print(f"  Processed destination: {paths.processed_dir}")

    summary = parse_and_process_dataset(
        raw_dir=paths.raw_dir,
        output_dir=paths.processed_dir,
        config=cfg,
        is_fixture=args.fixture
    )

    pfx = "[FIXTURE - meaningless] " if args.fixture else ""
    print(f"\n{pfx}=== PARSING COMPLETE ===")
    print(f"Total domains kept: {summary['final_counts']['total']} (Malware: {summary['final_counts']['malware']}, Benign: {summary['final_counts']['benign']})")
    print(f"Cross-class collisions dropped: {summary['cross_class_dropped_count']}")


if __name__ == "__main__":
    main()
