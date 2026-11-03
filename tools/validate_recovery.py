#!/usr/bin/env python3
"""Read-only structural validator for Malicious-HDG recovery artifacts.

This tool does not train, evaluate, recover, or write project artifacts.
It prints a JSON validation report to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SEEDS = (42, 1337, 2024, 777, 999)
SPLITS = ("random", "time", "group")
VARIANTS = ("sage", "attn")


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            return None, "JSON root is not an object"
        return value, None
    except FileNotFoundError:
        return None, "file missing"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read JSON: {type(exc).__name__}: {exc}"


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def sha256_labels(values: Any) -> str:
    # Match the project manifest convention: SHA-256 over int64 label bytes.
    import numpy as np
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_predictions(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    predictions = result.get("predictions")
    if not isinstance(predictions, dict):
        return ["per-sample predictions object is missing"]

    required = ("y_true", "y_prob", "y_pred")
    for key in required:
        if not isinstance(predictions.get(key), list):
            errors.append(f"predictions.{key} is missing or is not a list")

    if errors:
        return errors

    lengths = {key: len(predictions[key]) for key in required}
    if len(set(lengths.values())) != 1:
        errors.append(f"prediction array lengths differ: {lengths}")
        return errors

    count = lengths["y_true"]
    if count == 0:
        errors.append("prediction arrays are empty")
        return errors

    if any(not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1)
           for v in predictions["y_true"]):
        errors.append("y_true contains values other than integer 0 or 1")
    if any(not isinstance(v, int) or isinstance(v, bool) or v not in (0, 1)
           for v in predictions["y_pred"]):
        errors.append("y_pred contains values other than integer 0 or 1")
    if any(not finite_number(v) or not 0 <= v <= 1
           for v in predictions["y_prob"]):
        errors.append("y_prob contains non-finite values or values outside [0, 1]")

    metrics = result.get("metrics")
    test_counts = result.get("test_counts")
    if test_counts is None and isinstance(metrics, dict):
        test_counts = metrics.get("test_counts")

    if test_counts is not None:
        if not isinstance(test_counts, dict):
            errors.append("test_counts is not an object")
        else:
            declared = test_counts.get("total")
            if (not isinstance(declared, int) or isinstance(declared, bool)
                    or declared < 0):
                errors.append("test_counts.total missing or invalid")
            elif declared != count:
                errors.append(
                    f"test_counts.total={declared} does not match prediction length={count}"
                )

    for key in ("sample_ids", "test_ids", "test_indices"):
        if key not in predictions:
            continue

        values = predictions[key]
        if not isinstance(values, list):
            errors.append(f"predictions.{key} is not a list")
            continue
        if len(values) != count:
            errors.append(
                f"predictions.{key} length={len(values)} does not match prediction length={count}"
            )
            continue

        if key == "test_indices":
            valid = all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
        else:
            valid = all(
                (isinstance(value, str) and bool(value.strip()))
                or (isinstance(value, int) and not isinstance(value, bool))
                for value in values
            )

        if not valid:
            errors.append(f"predictions.{key} contains invalid identifiers")
        elif len(set(values)) != len(values):
            errors.append(f"predictions.{key} contains duplicate identifiers")

    return errors


def inspect_checkpoint(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        import torch
        # weights_only=True avoids arbitrary pickle object loading on supported
        # PyTorch versions. Do not fall back to unsafe pickle loading.
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return None, "installed PyTorch lacks weights_only loading; checkpoint not opened"
        if not isinstance(checkpoint, dict):
            return None, "checkpoint root is not a dictionary"
        return checkpoint, None
    except FileNotFoundError:
        return None, "file missing"
    except Exception as exc:
        return None, f"safe checkpoint load failed: {type(exc).__name__}: {exc}"


def validate_checkpoint(repo: Path, split: str, variant: str, seed: int, domain_count: int | None = None) -> dict[str, Any]:
    stem = f"gnn_{variant}_{split}_seed_{seed}"
    checkpoint_path = repo / "artifacts/checkpoints/real" / f"{stem}.pt"
    manifest_path = repo / "artifacts/checkpoints/real" / f"{stem}_manifest.json"
    errors: list[str] = []
    limitations: list[str] = []

    manifest, manifest_error = read_json(manifest_path)
    if manifest_error:
        errors.append(f"manifest: {manifest_error}")

    checkpoint, checkpoint_error = inspect_checkpoint(checkpoint_path)
    if checkpoint_error:
        message = f"checkpoint: {checkpoint_error}"
        if (
            checkpoint_error == "file missing"
            or checkpoint_error.startswith("installed PyTorch lacks weights_only")
            or checkpoint_error.startswith("safe checkpoint load failed:")
        ):
            limitations.append(message)
        else:
            errors.append(message)

    if manifest is not None:
        expected = {
            "architecture": variant,
            "split_type": split,
            "seed": seed,
            "checkpoint_file": checkpoint_path.name,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                errors.append(
                    f"manifest {key} mismatch: expected {value!r}, got {manifest.get(key)!r}"
                )

    if checkpoint is not None:
        expected = {"split_type": split, "seed": seed}
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                errors.append(
                    f"checkpoint {key} mismatch: expected {value!r}, got {checkpoint.get(key)!r}"
                )

        config = checkpoint.get("model_config")
        if not isinstance(config, dict):
            errors.append("checkpoint model_config missing or invalid")
        elif config.get("variant") != variant:
            errors.append("checkpoint model_config.variant mismatch")

        split_indices = {}
        for key in ("val_indices", "cert_indices", "test_indices"):
            indices = checkpoint.get(key)
            if not isinstance(indices, (list, tuple)):
                errors.append(f"checkpoint {key} missing or invalid")
                continue
            if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
                errors.append(f"checkpoint {key} contains invalid indices")
                continue
            if len(indices) != len(set(indices)):
                errors.append(f"checkpoint {key} contains duplicate indices")
            split_indices[key] = set(indices)
            if domain_count is not None and any(i >= domain_count for i in indices):
                errors.append(
                    f"checkpoint {key} contains index outside dataset bounds "
                    f"(domain_count={domain_count})"
                )

        if domain_count is None:
            limitations.append("checkpoint index bounds not verified; valid domain_count unavailable")

        split_names = tuple(split_indices)
        for i, first in enumerate(split_names):
            for second in split_names[i + 1:]:
                overlap = split_indices[first] & split_indices[second]
                if overlap:
                    errors.append(
                        f"checkpoint {first} and {second} overlap "
                        f"({len(overlap)} indices)"
                    )

        if "train_indices" not in checkpoint:
            limitations.append("checkpoint does not store train_indices")
        if not any("scaler" in str(k).lower() for k in checkpoint):
            limitations.append("checkpoint does not expose saved preprocessing scaler state")

    if manifest is not None and checkpoint is not None:
        for key in ("experiment_fingerprint", "seed", "split_type", "threshold"):
            if key in manifest and key in checkpoint:
                if manifest[key] != checkpoint[key]:
                    errors.append(f"manifest/checkpoint {key} mismatch")
        manifest_fp = manifest.get("experiment_fingerprint")
        checkpoint_fp = checkpoint.get("experiment_fingerprint")
        if not isinstance(manifest_fp, str) or not manifest_fp.strip():
            errors.append("manifest experiment_fingerprint missing or invalid")
        if not isinstance(checkpoint_fp, str) or not checkpoint_fp.strip():
            errors.append("checkpoint experiment_fingerprint missing or invalid")
        if manifest.get("model_config") != checkpoint.get("model_config"):
            errors.append("manifest/checkpoint model_config mismatch")

        for manifest_key, checkpoint_key in (
            ("val_count", "val_indices"),
            ("cert_count", "cert_indices"),
            ("test_count", "test_indices"),
        ):
            indices = checkpoint.get(checkpoint_key)
            if isinstance(indices, (list, tuple)):
                if manifest.get(manifest_key) != len(indices):
                    errors.append(f"manifest {manifest_key} does not match checkpoint {checkpoint_key}")

    # Missing expected files mean incomplete evidence, not proven corruption.
    missing = [item for item in errors if item.endswith(": file missing")]
    errors = [item for item in errors if item not in missing]
    limitations.extend(missing)

    return {
        "artifact": stem,
        "status": "BLOCKED" if errors else ("INCOMPLETE" if limitations else "ELIGIBLE"),
        "errors": errors,
        "limitations": limitations,
        "checkpoint_exists": checkpoint_path.is_file(),
        "manifest_exists": manifest_path.is_file(),
    }


def validate_gnn_result(repo: Path, split: str, variant: str, seed: int) -> dict[str, Any]:
    name = f"gnn_{variant}_{split}_seed_{seed}.json"
    path = repo / "results/real/runs" / name
    result, error = read_json(path)
    errors: list[str] = []
    limitations: list[str] = []

    if error:
        (limitations if error == "file missing" else errors).append(error)
    else:
        for key, expected in (("experiment", "gnn_evaluation"), ("variant", variant),
                              ("split_type", split), ("seed", seed)):
            if result.get(key) != expected:
                errors.append(f"{key} mismatch: expected {expected!r}, got {result.get(key)!r}")
        errors.extend(check_predictions(result))
        predictions = result.get("predictions")
        if isinstance(predictions, dict):
            if not any(k in predictions for k in ("sample_ids", "test_ids", "test_indices")):
                limitations.append("prediction arrays have no sample IDs or test indices")

    return {
        "artifact": name,
        "status": "BLOCKED" if errors else ("INCOMPLETE" if limitations else "ELIGIBLE"),
        "errors": errors,
        "limitations": limitations,
    }


def validate_baseline_result(repo: Path, baseline: str, seed: int) -> dict[str, Any]:
    name = f"baseline_xgboost_{baseline}_seed_{seed}.json"
    path = repo / "results/real/runs" / name
    result, error = read_json(path)
    errors: list[str] = []
    limitations: list[str] = []

    if error:
        (limitations if error == "file missing" else errors).append(error)
    else:
        for key in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            if not finite_number(result.get(key)):
                errors.append(f"metric {key} missing or non-finite")
        counts = result.get("test_counts")
        total = counts.get("total") if isinstance(counts, dict) else None
        if (
            not isinstance(total, int)
            or isinstance(total, bool)
            or total < 0
        ):
            errors.append("test_counts.total missing or invalid")
        if not isinstance(result.get("predictions"), dict):
            limitations.append("no per-sample prediction arrays; paired comparison unavailable")
        else:
            errors.extend(check_predictions(result))
            limitations.append("baseline prediction sample IDs are not established")

    return {
        "artifact": name,
        "status": "BLOCKED" if errors else ("INCOMPLETE" if limitations else "ELIGIBLE"),
        "errors": errors,
        "limitations": limitations,
    }


def validate_dataset(repo: Path) -> dict[str, Any]:
    processed = repo / "data/processed/real"
    manifest_path = processed / "graph_manifest.json"
    manifest, error = read_json(manifest_path)
    errors: list[str] = []
    limitations: list[str] = []

    if error:
        if error == "file missing":
            limitations.append("graph manifest: file missing")
            return {"status": "INCOMPLETE", "errors": errors, "limitations": limitations}
        errors.append(f"graph manifest: {error}")
        return {"status": "BLOCKED", "errors": errors, "limitations": limitations}

    cohort_path = processed / "domains_cohort.parquet"
    if not cohort_path.is_file():
        limitations.append("domains_cohort.parquet missing; cannot verify ordered cohort identity")
    else:
        try:
            import pandas as pd
            frame = pd.read_parquet(cohort_path)
            if "domain" not in frame.columns or "label" not in frame.columns:
                errors.append("cohort parquet must contain domain and label columns")
            else:
                if manifest.get("domain_count") != len(frame):
                    errors.append("manifest domain_count does not match cohort row count")
                names_hash = sha256_text(frame["domain"].astype(str).tolist())
                if manifest.get("domain_names_sha256") != names_hash:
                    errors.append("manifest domain_names_sha256 does not match ordered cohort")
                labels_hash = sha256_labels(frame["label"].to_numpy())
                if manifest.get("labels_sha256") != labels_hash:
                    errors.append("manifest labels_sha256 does not match cohort labels")
        except Exception as exc:
            limitations.append(f"cohort could not be inspected: {type(exc).__name__}: {exc}")

    graph_path = processed / "heterodata.pt"
    if not graph_path.is_file():
        limitations.append("heterodata.pt missing; graph integrity could not be checked")
    else:
        try:
            import torch
            try:
                graph = torch.load(graph_path, map_location="cpu", weights_only=True)
            except TypeError:
                graph = None
                limitations.append("PyTorch lacks weights_only loading; graph file not opened")
            if graph is not None:
                domain_count = int(graph["domain"].num_nodes)
                if manifest.get("domain_count") != domain_count:
                    errors.append("graph domain node count does not match manifest domain_count")
        except Exception as exc:
            limitations.append(
                f"graph exists but safe inspection was unavailable: "
                f"{type(exc).__name__}: {exc}"
            )

    if not manifest.get("manifest_id"):
        limitations.append("manifest_id missing")
    return {
        "status": "BLOCKED" if errors else ("INCOMPLETE" if limitations else "ELIGIBLE"),
        "errors": errors,
        "limitations": limitations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-dataset", action="store_true",
                        help="skip parquet/graph integrity checks")
    args = parser.parse_args()
    repo = args.repo.resolve()

    domain_count = None
    if not args.skip_dataset:
        dataset_manifest, _ = read_json(repo / "data/processed/real/graph_manifest.json")
        if isinstance(dataset_manifest, dict):
            candidate = dataset_manifest.get("domain_count")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                domain_count = candidate

    report: dict[str, Any] = {
        "tool": "validate_recovery",
        "read_only": True,
        "repository": str(repo),
        "scope": "structural validation only; no evaluation or recovery",
        "dataset": None if args.skip_dataset else validate_dataset(repo),
        "checkpoints": [],
        "gnn_results": [],
        "baseline_results": [],
        "comparison_limitations": [
            "equal prediction-array lengths do not prove identical sample ordering",
            "sample-level paired comparisons require verified common sample IDs",
            "checkpoint metadata lacks train_indices and saved scaler state",
        ],
    }

    for split in SPLITS:
        for variant in VARIANTS:
            for seed in SEEDS:
                report["checkpoints"].append(validate_checkpoint(repo, split, variant, seed, domain_count))
                report["gnn_results"].append(validate_gnn_result(repo, split, variant, seed))

    for baseline in ("tabular", "lexical"):
        for seed in SEEDS:
            report["baseline_results"].append(validate_baseline_result(repo, baseline, seed))

    records = report["checkpoints"] + report["gnn_results"] + report["baseline_results"]
    if report["dataset"] is not None:
        records.append(report["dataset"])
    counts = {status: sum(r["status"] == status for r in records)
              for status in ("ELIGIBLE", "INCOMPLETE", "BLOCKED")}
    report["summary"] = {
        "counts": counts,
        "overall_status": (
            "BLOCKED" if counts["BLOCKED"] else
            "INCOMPLETE" if counts["INCOMPLETE"] else "ELIGIBLE"
        ),
        "checkpoint_count_expected": 30,
        "gnn_result_count_expected": 30,
        "baseline_result_count_expected": 10,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if counts["BLOCKED"] else 0


if __name__ == "__main__":
    sys.exit(main())
