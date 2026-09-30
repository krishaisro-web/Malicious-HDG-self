#!/usr/bin/env python3
"""
CLI script to generate schema-faithful synthetic fixtures for Zenodo DomainRadar v2.
Outputs strictly to data_fixture/zenodo/.
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hdg.fixture import make_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic schema-faithful fixture.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_fixture/zenodo",
        help="Target output directory for fixture files"
    )
    parser.add_argument("--n-malware", type=int, default=3000, help="Number of malware records")
    parser.add_argument("--n-umbrella", type=int, default=1500, help="Number of Umbrella benign records")
    parser.add_argument("--n-cesnet", type=int, default=1500, help="Number of CESNET benign records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    print(f"[FIXTURE] Generating schema-faithful fixture at: {out_path}...")
    files = make_fixture(
        output_dir=out_path,
        n_malware=args.n_malware,
        n_umbrella=args.n_umbrella,
        n_cesnet=args.n_cesnet,
        seed=args.seed
    )
    for k, p in files.items():
        size_kb = p.stat().st_size / 1024
        print(f"  - {k}: {p} ({size_kb:.1f} KB)")
    print("[FIXTURE] Generation complete.")


if __name__ == "__main__":
    main()
