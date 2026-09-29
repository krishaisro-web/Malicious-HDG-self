from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data_processed" / "chrmor" / "chrmor_clean.csv"
OUTPUT_DIR = ROOT / "results" / "chrmor"

METRICS_FILE = OUTPUT_DIR / "lexical_baseline_metrics.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "lexical_baseline_test_predictions.csv"
FAMILY_FILE = OUTPUT_DIR / "lexical_baseline_family_metrics.csv"

RANDOM_STATE = 42


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, dtype=str)

    expected_columns = ["label", "family", "domain"]
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns: {list(df.columns)}; "
            f"expected: {expected_columns}"
        )

    if df[["label", "domain"]].isnull().any().any():
        raise ValueError("Null label/domain values found.")

    label_map = {"legit": 0, "dga": 1}
    unknown_labels = sorted(set(df["label"]) - set(label_map))
    if unknown_labels:
        raise ValueError(f"Unexpected labels: {unknown_labels}")

    X = df["domain"].str.lower().str.rstrip(".")
    y = df["label"].map(label_map).astype(int)

    print(f"Total domains: {len(df):,}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    # Same stratified 70/15/15 split as the original experiment.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    print("\nSplit sizes:")
    print(f"train: {len(X_train):,}")
    print(f"val:   {len(X_val):,}")
    print(f"test:  {len(X_test):,}")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print("\nFeature matrix:")
    print(f"train: {X_train_vec.shape}")
    print(f"val:   {X_val_vec.shape}")
    print(f"test:  {X_test_vec.shape}")

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_vec, y_train)

    def evaluate(name, X_vec, y_true):
        probabilities = model.predict_proba(X_vec)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        return {
            "split": name,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_true, probabilities),
            "pr_auc": average_precision_score(y_true, probabilities),
        }

    results = [
        evaluate("validation", X_val_vec, y_val),
        evaluate("test", X_test_vec, y_test),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_FILE, index=False)

    # Save test predictions with the original domain and DGA family.
    test_positions = X_test.index

    test_probabilities = model.predict_proba(X_test_vec)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    test_output = df.loc[test_positions, ["domain", "family", "label"]].copy()
    test_output["true_label"] = test_output["label"].map(label_map)
    test_output["probability_dga"] = test_probabilities
    test_output["prediction"] = test_predictions

    test_output = test_output[
        [
            "domain",
            "family",
            "label",
            "true_label",
            "probability_dga",
            "prediction",
        ]
    ]

    test_output.to_csv(PREDICTIONS_FILE, index=False)

    # Family-level evaluation for DGA families only.
    family_rows = []

    for family in sorted(
        test_output.loc[test_output["label"] == "dga", "family"].unique()
    ):
        family_df = test_output[test_output["family"] == family]

        # Every row in a DGA family is a positive-class example.
        family_true = family_df["true_label"].to_numpy()
        family_pred = family_df["prediction"].to_numpy()

        family_rows.append(
            {
                "family": family,
                "test_samples": len(family_df),
                "recall": recall_score(
                    family_true,
                    family_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    family_true,
                    family_pred,
                    zero_division=0,
                ),
            }
        )

    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(FAMILY_FILE, index=False)

    print("\nResults:")
    print(results_df.to_string(index=False))

    print("\nDGA family results:")
    print(family_df.to_string(index=False))

    print(f"\nSaved: {METRICS_FILE}")
    print(f"Saved: {PREDICTIONS_FILE}")
    print(f"Saved: {FAMILY_FILE}")


if __name__ == "__main__":
    main()
