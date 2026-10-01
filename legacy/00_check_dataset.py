from pathlib import Path

REQUIRED_FILES = [
    Path("data_raw/zenodo/malware.json"),
    Path("data_raw/zenodo/benign_umbrella.json"),
]

missing = [str(p) for p in REQUIRED_FILES if not p.exists()]

if missing:
    print("Missing required dataset files:")
    for f in missing:
        print(" -", f)
    print("\nPlace the raw dataset files in data_raw/zenodo/ before running the parser.")
    raise SystemExit(1)

print("Dataset files found:")
for f in REQUIRED_FILES:
    print(" -", f)
