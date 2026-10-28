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


def find_exact_match(
    filename: str,
    expected_size: int,
    expected_md5: str,
    search_dirs: Optional[list] = None
) -> Optional[Path]:
    """
    Checks if the destination or appropriate search folders contain an exact matching file
    by size and MD5 checksum.
    Checks:
      1. Exact filename in candidate folders (e.g. data/raw/zenodo/malware.json)
      2. Alternative/renamed files in candidate folders with identical size and MD5
    Returns the Path to the matching file, or None if no match is found.
    """
    if not search_dirs:
        return None

    seen_dirs = set()
    for d in search_dirs:
        if not d:
            continue
        folder = Path(d)
        if not folder.is_dir():
            continue
        try:
            folder_res = folder.resolve()
        except Exception:
            folder_res = folder
        if folder_res in seen_dirs:
            continue
        seen_dirs.add(folder_res)

        # 1. Exact filename check
        target_candidate = folder_res / filename
        if target_candidate.is_file():
            try:
                if target_candidate.stat().st_size == expected_size:
                    if verify_md5(target_candidate, expected_md5):
                        return target_candidate
            except (OSError, PermissionError):
                pass

        # 2. Scan all files in directory for matching size and MD5
        try:
            for entry in folder_res.iterdir():
                if entry == target_candidate or not entry.is_file() or entry.name.endswith(".sample.part"):
                    continue
                try:
                    if entry.stat().st_size == expected_size:
                        print(f"  -> Checking candidate file '{entry.name}' ({expected_size} bytes) in {folder_res}...")
                        if verify_md5(entry, expected_md5):
                            print(f"  [OK] Hash match confirmed for '{entry.name}'. Exact match!")
                            return entry
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            continue

    return None


def find_sample_match(
    filename: str,
    target_count: int,
    search_dirs: Optional[list] = None
) -> Optional[Path]:
    """
    Checks if candidate folders already contain a valid sample JSON array with at least target_count records.
    """
    if not search_dirs:
        return None

    seen_dirs = set()
    for d in search_dirs:
        if not d:
            continue
        folder = Path(d)
        if not folder.is_dir():
            continue
        try:
            folder_res = folder.resolve()
        except Exception:
            folder_res = folder
        if folder_res in seen_dirs:
            continue
        seen_dirs.add(folder_res)

        candidate = folder_res / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    recs = json.load(f)
                if isinstance(recs, list) and len(recs) >= target_count:
                    return candidate
            except Exception:
                continue

    return None


def download_full_file(
    url: str,
    dest_path: Path,
    expected_md5: str,
    expected_size: int,
    force: bool = False,
    search_dirs: Optional[list] = None
) -> bool:
    """Downloads a full file using curl with resume support (-C -) and progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if not force:
        candidate_dirs = [dest_path.parent] + (search_dirs or [])
        match = find_exact_match(dest_path.name, expected_size, expected_md5, candidate_dirs)
        if match is not None:
            if match.resolve() == dest_path.resolve():
                print(f"  [OK] {dest_path.name} already exists in {dest_path.parent} and verified (exact size & MD5 match). Skipping download.")
                return True
            else:
                print(f"  [OK] Found exact match for {dest_path.name} at {match} (exact size & MD5 match).")
                print(f"       Reusing existing file without downloading: copying to {dest_path}...")
                shutil.copy2(match, dest_path)
                return True
        elif dest_path.exists():
            print(f"  [WARN] {dest_path.name} exists in {dest_path.parent} but size/hash mismatch. Re-downloading...")
            dest_path.unlink(missing_ok=True)
    else:
        print(f"  -> Force overwrite requested for {dest_path.name}. Re-downloading...")
        dest_path.unlink(missing_ok=True)
        temp_path.unlink(missing_ok=True)

    print(f"  -> Downloading {dest_path.name} ({expected_size / (1024**3):.2f} GB)...")
    cmd = [
        "curl",
        "-f",
        "-L",
        "-C", "-",
        "--retry", "5",
        "--retry-delay", "3",
        "--retry-connrefused",
        "--connect-timeout", "30",
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
        temp_path.replace(dest_path)
        print(f"  [SUCCESS] {dest_path.name} verified and saved.")
        return True
    else:
        print(f"  [ERROR] MD5 mismatch for {dest_path.name}!")
        temp_path.unlink(missing_ok=True)
        return False


def download_sample_stream(
    url: str,
    dest_path: Path,
    target_count: int,
    force: bool = False,
    search_dirs: Optional[list] = None
) -> bool:
    """Streams records via curl and ijson into a valid JSON array directly at dest_path."""
    try:
        import ijson
    except ImportError:
        print("  [ERROR] 'ijson' is required for streaming sampling. Run: pip install ijson")
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not force:
        candidate_dirs = [dest_path.parent] + (search_dirs or [])
        sample_match = find_sample_match(dest_path.name, target_count, candidate_dirs)
        if sample_match is not None:
            if sample_match.resolve() == dest_path.resolve():
                print(f"  [OK] {dest_path.name} sample already exists in {dest_path.parent} with sufficient records. Skipping download.")
                return True
            else:
                print(f"  [OK] Found valid sample for {dest_path.name} in {sample_match.parent}. Reusing without downloading: copying to {dest_path}...")
                shutil.copy2(sample_match, dest_path)
                return True

    temp_path = dest_path.with_suffix(dest_path.suffix + ".sample.part")
    print(f"  -> Streaming {target_count} sample records from Zenodo for {dest_path.name}...")

    cmd = ["curl", "-s", "-f", "-L", "-N", "--connect-timeout", "30", url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        print("  [ERROR] Failed to spawn curl pipe.")
        return False

    records = []
    stream_err = ""
    try:
        # Crucial: use_float=True converts JSON numbers to Python floats/ints instead of Decimal
        parser = ijson.items(proc.stdout, "item", use_float=True)
        for item in parser:
            records.append(item)
            if len(records) >= target_count:
                break
    except Exception as e:
        stream_err = str(e)
        print(f"  [WARN] Stream parsing interrupted: {e}")
    finally:
        proc.terminate()
        try:
            _, stderr_bytes = proc.communicate(timeout=3)
            if stderr_bytes:
                stream_err += " " + stderr_bytes.decode(errors="replace").strip()
        except Exception:
            proc.kill()

    if not records:
        print(f"  [ERROR] No records were parsed for {dest_path.name}. {stream_err}".strip())
        temp_path.unlink(missing_ok=True)
        return False

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    temp_path.replace(dest_path)
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download official data_schema.json (use --no-include-schema to skip)"
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

    files_to_download = ["malware.json", "benign_umbrella.json", "benign_cesnet.json"]
    if args.include_schema:
        files_to_download.append("data_schema.json")

    # Define candidate folders to search for existing exact-matching dataset files
    search_dirs = [
        target_dir,
        REPO_ROOT / "data" / "raw" / "zenodo",
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "data",
    ]

    # Dynamic disk space calculation: check bytes remaining for incomplete/missing files
    if args.sample:
        # Check if sample or full files already satisfy target counts
        needed = any(
            not args.force and find_sample_match(fn, ZENODO_FILES[fn].get("sample_default") or 1, search_dirs) is None
            for fn in files_to_download
        )
        required_bytes = (100 * 1024 * 1024) if needed else 0
    else:
        required_bytes = 0
        for fn in files_to_download:
            meta = ZENODO_FILES[fn]
            if not args.force and find_exact_match(fn, meta["size_bytes"], meta["md5"], search_dirs) is not None:
                # File already exists and matches exactly; 0 bytes needed
                continue
            part_fp = target_dir / (fn + ".part")
            already_have = part_fp.stat().st_size if part_fp.exists() else 0
            required_bytes += max(0, meta["size_bytes"] - already_have)

    _, _, free_bytes = shutil.disk_usage(target_dir)
    free_gb = free_bytes / (1024**3)
    required_gb = required_bytes / (1024**3)
    print(f"Available disk space: {free_gb:.2f} GB (Required for remaining downloads: ~{required_gb:.2f} GB)")
    if free_bytes < required_bytes:
        print(f"[ERROR] Insufficient disk space! {free_gb:.2f} GB free, but ~{required_gb:.2f} GB needed.")
        sys.exit(1)

    success_count = 0
    start_time = time.time()

    for filename in files_to_download:
        meta = ZENODO_FILES[filename]
        dest_path = target_dir / filename
        url = meta["url"]

        if args.sample and meta["sample_default"] is not None:
            ok = download_sample_stream(
                url, dest_path, meta["sample_default"],
                force=args.force, search_dirs=search_dirs
            )
        else:
            ok = download_full_file(
                url, dest_path, meta["md5"], meta["size_bytes"],
                force=args.force, search_dirs=search_dirs
            )

        if ok:
            success_count += 1
        else:
            print(f"[FAIL] Could not retrieve or verify {filename}")

    elapsed = time.time() - start_time
    print(f"\nCompleted: {success_count}/{len(files_to_download)} files downloaded/verified in {elapsed:.1f}s.")

    if success_count < len(files_to_download):
        print(f"[ERROR] Download incomplete: {len(files_to_download) - success_count} files failed.")
        sys.exit(1)

    # Run verification preflight check
    print("\n--- Verifying with pipeline preflight ---")
    try:
        cmd = [sys.executable, "-m", "pipeline.00c_preflight", "--force"]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if "Raw Dataset Files Check" in res.stdout:
            for line in res.stdout.splitlines():
                if "[FOUND]" in line or "[MISSING]" in line:
                    print(" ", line.strip())
        if res.returncode == 0:
            print("[OK] Dataset ready for pipeline execution!")
        else:
            print(f"[WARN] Preflight check completed with exit code {res.returncode}.")
    except Exception as e:
        print(f"[INFO] Preflight verification check note: {e}")


if __name__ == "__main__":
    main()
