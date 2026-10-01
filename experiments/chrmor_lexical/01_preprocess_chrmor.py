from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT / "data" / "raw" / "chrmor" / "dga_domains_full.csv"
OUT_DIR = ROOT / "data" / "processed" / "chrmor"
OUT_FILE = OUT_DIR / "chrmor_clean.csv"

EXPECTED_COLUMNS = ["label", "family", "domain"]
VALID_LABELS = {"dga", "legit"}


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        RAW_FILE,
        header=None,
        names=EXPECTED_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )

    print(f"Raw rows: {len(df):,}")

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    if df.isnull().any().any():
        raise ValueError("Unexpected null values found.")

    invalid_labels = sorted(set(df["label"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(f"Unexpected labels: {invalid_labels}")

    empty_domains = int((df["domain"].str.strip() == "").sum())
    if empty_domains:
        raise ValueError(f"Empty domains found: {empty_domains}")

    conflicting = (
        df.groupby("domain")["label"]
        .nunique()
        .loc[lambda s: s > 1]
    )

    if len(conflicting):
        raise ValueError(
            f"Found {len(conflicting)} domains with conflicting labels."
        )

    duplicate_rows = int(df.duplicated(subset=["domain"], keep="first").sum())

    df = df.drop_duplicates(subset=["domain"], keep="first").reset_index(drop=True)

    print(f"Duplicate rows removed: {duplicate_rows:,}")
    print(f"Clean unique domains: {len(df):,}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print("\nDGA family distribution:")
    print(df.loc[df["label"] == "dga", "family"].value_counts().to_string())

    df.to_csv(OUT_FILE, index=False)

    print(f"\nSaved: {OUT_FILE}")


if __name__ == "__main__":
    main()
