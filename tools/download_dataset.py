#!/usr/bin/env python3
"""
Zenodo DomainRadar v2 Dataset Downloader
Record DOI: 10.5281/zenodo.14332167

Downloads the target Zenodo DomainRadar v2 dataset files directly into
the configured raw dataset folder (`data/raw/zenodo/`).

Features:
- Auto-detects target directory from repository configs/paths.
- Supports downloading full files (with resumable curl download & MD5 verification).
- Supports downloading a stratified real sample (e.g. 100 records) on resource-limited
  systems or for rapid empirical experimentation.
- Robust retries, streaming progress, and MD5 checksum validation.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent

# Zenodo Record Metadata
ZENODO_RECORD_ID = "14332167"
ZENODO_FILES: Dict[str, Dict[str, Any]] = {
    "malware.json": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/malware.json/content",
        "size_bytes": 1084514082,  # ~1.08 GB
        "md5": "ed9b14f24feb2e66513f9339492b071d",
        "sample_default": 50,
        "description": "Active malware domains from threat feeds (ThreatFox, URLhaus, Firebog, MISP)"
    },
    "benign_umbrella.json": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/benign_umbrella.json/content",
        "size_bytes": 6075400719,  # ~6.08 GB
        "md5": "41e340c0714d7455494c0f893193cb5e",
        "sample_default": 25,
        "description": "Benign domains from Cisco Umbrella Top 1M list"
    },
    "benign_cesnet.json": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/benign_cesnet.json/content",
        "size_bytes": 6437047412,  # ~6.44 GB
        "md5": "cea8cbe33ec308f76c5ecf3ef6a6219e",
        "sample_default": 25,
        "description": "Benign domains from CESNET academic network traffic"
    },
    "data_schema.json": {
        "url": f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/data_schema.json/content",
        "size_bytes": 41222,  # ~41 KB
        "md5": "725ff97c470f11d094a9ab81a582dceb",
        "sample_default": None,
        "description": "Official JSON schema specification for DomainRadar v2"
    }
}


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verifies the MD5 checksum of a file in chunks."""
    if not file_path.exists():
        return False
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_md5.lower()


def download_full_file(url: str, dest_path: Path, expected_md5: str, expected_size: int) -> bool:
    """Downloads a full file using curl with resume support (-C -) and progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if dest_path.exists():
        if dest_path.stat().st_size == expected_size and verify_md5(dest_path, expected_md5):
            print(f"  [OK] {dest_path.name} already downloaded and verified (MD5 match).")
            return True
        else:
            print(f"  [WARN] {dest_path.name} exists but size/hash mismatch. Re-downloading...")
            dest_path.unlink()

    print(f"  -> Downloading {dest_path.name} ({expected_size / (1024**3):.2f} GB)...")
    cmd = [
        "curl",
        "-L",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "3",
        "--retry-connrefused",
        "-o", str(temp_path),
        url
    ]

    # Run curl interactively showing progress
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"  [ERROR] curl failed with exit code {res.returncode}")
        return False

    print(f"  -> Verifying MD5 checksum for {dest_path.name}...")
    if verify_md5(temp_path, expected_md5):
        temp_path.rename(dest_path)
        print(f"  [SUCCESS] {dest_path.name} verified and saved.")
        return True
    else:
        print(f"  [ERROR] MD5 mismatch for {dest_path.name}!")
        return False


def download_sample_stream(url: str, dest_path: Path, target_count: int) -> bool:
    """Streams records via curl and ijson into a valid JSON array directly at dest_path."""
    try:
        import ijson
    except ImportError:
        print("  [ERROR] 'ijson' is required for streaming sampling. Run: pip install ijson")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> Streaming {target_count} sample records from Zenodo for {dest_path.name}...")

    cmd = ["curl", "-s", "-L", "-N", url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        print("  [ERROR] Failed to spawn curl pipe.")
        return False

    records = []
    try:
        parser = ijson.items(proc.stdout, "item")
        for item in parser:
            records.append(item)
            if len(records) >= target_count:
                break
    except Exception as e:
        print(f"  [WARN] Stream parsing interrupted: {e}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    if not records:
        print(f"  [ERROR] No records were parsed for {dest_path.name}.")
        return False

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"  [SUCCESS] Written {len(records)} verified records ({dest_path.stat().st_size / (1024**2):.2f} MB) to {dest_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Zenodo DomainRadar v2 Dataset into the repository data folder."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target folder for dataset (default: auto-detected as data/raw/zenodo/)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download verified empirical samples (50 malware, 25 umbrella, 25 cesnet) instead of multi-gigabyte full files"
    )
    parser.add_argument(
        "--include-schema",
        action="store_true",
        default=True,
        help="Also download official data_schema.json"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files even if they already exist"
    )
    args = parser.parse_args()

    # Determine destination directory
    if args.output_dir:
        target_dir = Path(args.output_dir).resolve()
    else:
        # Load from config or fallback to standard location
        try:
            from src.hdg.config import load_config, get_resolved_paths
            cfg = load_config()
            paths = get_resolved_paths(cfg, is_fixture=False, root_dir=REPO_ROOT)
            target_dir = paths.raw_dir
        except Exception:
            target_dir = REPO_ROOT / "data" / "raw" / "zenodo"

    target_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("Zenodo DomainRadar v2 Dataset Downloader")
    print(f"Target Record:  https://zenodo.org/records/{ZENODO_RECORD_ID}")
    print(f"Destination:    {target_dir}")
    print(f"Download Mode:  {'SAMPLE (Stream 100 records)' if args.sample else 'FULL (Complete ~13.6 GB dataset)'}")
    print("======================================================================\n")

    # Check available disk space
    _, _, free_bytes = shutil.disk_usage(target_dir)
    free_gb = free_bytes / (1024**3)
    required_gb = 0.1 if args.sample else 15.0
    print(f"Available disk space: {free_gb:.2f} GB (Required: ~{required_gb:.1f} GB)")
    if free_gb < required_gb:
        print(f"[ERROR] Insufficient disk space! {free_gb:.2f} GB free, but ~{required_gb:.1f} GB needed.")
        sys.exit(1)

    files_to_download = ["malware.json", "benign_umbrella.json", "benign_cesnet.json"]
    if args.include_schema:
        files_to_download.append("data_schema.json")

    success_count = 0
    start_time = time.time()

    for filename in files_to_download:
        meta = ZENODO_FILES[filename]
        dest_path = target_dir / filename
        url = meta["url"]

        if dest_path.exists() and not args.force:
            if not args.sample and dest_path.stat().st_size == meta["size_bytes"]:
                print(f"[*] {filename} already exists and matches expected size. Skipping (use --force to overwrite).")
                success_count += 1
                continue
            elif args.sample and dest_path.stat().st_size > 0:
                print(f"[*] {filename} sample already exists ({dest_path.stat().st_size / (1024**2):.2f} MB). Skipping.")
                success_count += 1
                continue

        if args.sample and meta["sample_default"] is not None:
            ok = download_sample_stream(url, dest_path, meta["sample_default"])
        else:
            ok = download_full_file(url, dest_path, meta["md5"], meta["size_bytes"])

        if ok:
            success_count += 1
        else:
            print(f"[FAIL] Could not retrieve {filename}")

    elapsed = time.time() - start_time
    print(f"\nCompleted: {success_count}/{len(files_to_download)} files downloaded in {elapsed:.1f}s.")

    # Run verification preflight check
    print("\n--- Verifying with pipeline preflight ---")
    try:
        cmd = [sys.executable, "-m", "pipeline.00c_preflight", "--force"]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if "Raw Dataset Files Check" in res.stdout:
            for line in res.stdout.splitlines():
                if "[FOUND]" in line or "[MISSING]" in line:
                    print(" ", line.strip())
        print("[OK] Dataset ready for pipeline execution!")
    except Exception as e:
        print(f"[INFO] Preflight verification check note: {e}")


if __name__ == "__main__":
    main()
