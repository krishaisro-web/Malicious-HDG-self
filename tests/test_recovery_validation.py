import json

from tools.validate_recovery import (
    check_predictions,
    read_json,
    sha256_text,
    validate_baseline_result,
)


def test_read_json_accepts_object(tmp_path):
    path = tmp_path / "valid.json"
    path.write_text('{"seed": 42}', encoding="utf-8")

    value, error = read_json(path)

    assert value == {"seed": 42}
    assert error is None


def test_read_json_reports_missing_file(tmp_path):
    value, error = read_json(tmp_path / "missing.json")

    assert value is None
    assert error == "file missing"


def test_read_json_rejects_non_object_json(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    value, error = read_json(path)

    assert value is None
    assert error == "JSON root is not an object"


def test_prediction_arrays_pass_when_consistent():
    result = {
        "predictions": {
            "y_true": [0, 1, 1],
            "y_prob": [0.1, 0.8, 0.9],
            "y_pred": [0, 1, 1],
        },
        "test_counts": {"total": 3},
    }

    assert check_predictions(result) == []


def test_prediction_arrays_fail_when_lengths_differ():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.1],
            "y_pred": [0, 1],
        }
    }

    errors = check_predictions(result)

    assert any("lengths differ" in error for error in errors)


def test_prediction_arrays_fail_for_invalid_probabilities():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.2, 1.5],
            "y_pred": [0, 1],
        }
    }

    errors = check_predictions(result)

    assert any("outside [0, 1]" in error for error in errors)


def test_prediction_count_must_match_declared_test_count():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "y_pred": [0, 1],
        },
        "test_counts": {"total": 99},
    }

    errors = check_predictions(result)

    assert any("does not match prediction length" in error for error in errors)


def test_baseline_without_predictions_is_incomplete_not_paired(tmp_path):
    runs = tmp_path / "results" / "real" / "runs"
    runs.mkdir(parents=True)
    result = {
        "roc_auc": 0.88,
        "pr_auc": 0.77,
        "f1": 0.74,
        "precision": 0.69,
        "recall": 0.80,
        "test_counts": {"total": 15001},
    }
    (runs / "baseline_xgboost_tabular_seed_42.json").write_text(
        json.dumps(result), encoding="utf-8"
    )

    checked = validate_baseline_result(tmp_path, "tabular", 42)

    assert checked["status"] == "INCOMPLETE"
    assert any("paired comparison unavailable" in item
               for item in checked["limitations"])


def test_sha256_text_is_order_sensitive():
    assert sha256_text(["a.example", "b.example"]) != sha256_text(
        ["b.example", "a.example"]
    )


def test_missing_baseline_is_incomplete(tmp_path):
    checked = validate_baseline_result(tmp_path, "tabular", 42)

    assert checked["status"] == "INCOMPLETE"
    assert "file missing" in checked["limitations"]


def test_missing_gnn_result_is_incomplete(tmp_path):
    from tools.validate_recovery import validate_gnn_result

    checked = validate_gnn_result(tmp_path, "random", "sage", 42)

    assert checked["status"] == "INCOMPLETE"
    assert "file missing" in checked["limitations"]


def test_nested_gnn_test_count_mismatch_is_detected():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "y_pred": [0, 1],
        },
        "metrics": {"test_counts": {"total": 99}},
    }

    errors = check_predictions(result)

    assert any("does not match prediction length" in error for error in errors)


def test_missing_dataset_manifest_is_incomplete(tmp_path):
    from tools.validate_recovery import validate_dataset

    checked = validate_dataset(tmp_path)

    assert checked["status"] == "INCOMPLETE"
    assert "graph manifest: file missing" in checked["limitations"]


def test_safe_checkpoint_load_failure_is_incomplete(tmp_path, monkeypatch):
    import tools.validate_recovery as validator

    monkeypatch.setattr(
        validator,
        "inspect_checkpoint",
        lambda path: (None, "safe checkpoint load failed: UnpicklingError"),
    )

    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "real"
    checkpoint_dir.mkdir(parents=True)
    manifest = {
        "architecture": "sage",
        "split_type": "random",
        "seed": 42,
        "checkpoint_file": "gnn_sage_random_seed_42.pt",
    }
    (checkpoint_dir / "gnn_sage_random_seed_42_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    checked = validator.validate_checkpoint(tmp_path, "random", "sage", 42)

    assert checked["status"] == "INCOMPLETE"
    assert any("safe checkpoint load failed" in item for item in checked["limitations"])


def test_missing_checkpoint_fingerprint_is_blocked(tmp_path, monkeypatch):
    import tools.validate_recovery as validator

    config = {"variant": "sage"}
    checkpoint = {
        "split_type": "random",
        "seed": 42,
        "model_config": config,
        "val_indices": [1],
        "cert_indices": [2],
        "test_indices": [3],
    }
    monkeypatch.setattr(
        validator,
        "inspect_checkpoint",
        lambda path: (checkpoint, None),
    )

    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "real"
    checkpoint_dir.mkdir(parents=True)
    manifest = {
        "architecture": "sage",
        "split_type": "random",
        "seed": 42,
        "checkpoint_file": "gnn_sage_random_seed_42.pt",
        "experiment_fingerprint": "expected-fingerprint",
        "model_config": config,
        "val_count": 1,
        "cert_count": 1,
        "test_count": 1,
    }
    (checkpoint_dir / "gnn_sage_random_seed_42_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    checked = validator.validate_checkpoint(tmp_path, "random", "sage", 42)

    assert checked["status"] == "BLOCKED"
    assert "checkpoint experiment_fingerprint missing or invalid" in checked["errors"]



def _run_checkpoint_validation_case(
    tmp_path, monkeypatch, checkpoint, manifest, domain_count=None
):
    import tools.validate_recovery as validator

    monkeypatch.setattr(
        validator,
        "inspect_checkpoint",
        lambda path: (checkpoint, None),
    )

    checkpoint_dir = tmp_path / "artifacts" / "checkpoints" / "real"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "gnn_sage_random_seed_42_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    return validator.validate_checkpoint(
        tmp_path, "random", "sage", 42, domain_count
    )


def _valid_checkpoint_and_manifest():
    config = {"variant": "sage"}
    fingerprint = "test-fingerprint"

    checkpoint = {
        "split_type": "random",
        "seed": 42,
        "model_config": config,
        "experiment_fingerprint": fingerprint,
        "val_indices": [1],
        "cert_indices": [2],
        "test_indices": [3],
    }
    manifest = {
        "architecture": "sage",
        "split_type": "random",
        "seed": 42,
        "checkpoint_file": "gnn_sage_random_seed_42.pt",
        "experiment_fingerprint": fingerprint,
        "model_config": config,
        "val_count": 1,
        "cert_count": 1,
        "test_count": 1,
    }
    return checkpoint, manifest


def test_checkpoint_duplicate_split_indices_are_blocked(tmp_path, monkeypatch):
    checkpoint, manifest = _valid_checkpoint_and_manifest()
    checkpoint["val_indices"] = [1, 1]
    manifest["val_count"] = 2

    checked = _run_checkpoint_validation_case(
        tmp_path, monkeypatch, checkpoint, manifest
    )

    assert checked["status"] == "BLOCKED"
    assert any("val_indices contains duplicate indices" in item
               for item in checked["errors"])


def test_checkpoint_overlapping_splits_are_blocked(tmp_path, monkeypatch):
    checkpoint, manifest = _valid_checkpoint_and_manifest()
    checkpoint["cert_indices"] = [1]

    checked = _run_checkpoint_validation_case(
        tmp_path, monkeypatch, checkpoint, manifest
    )

    assert checked["status"] == "BLOCKED"
    assert any("val_indices and cert_indices overlap" in item
               for item in checked["errors"])


def test_checkpoint_empty_fingerprints_are_blocked(tmp_path, monkeypatch):
    checkpoint, manifest = _valid_checkpoint_and_manifest()
    checkpoint["experiment_fingerprint"] = "   "
    manifest["experiment_fingerprint"] = ""

    checked = _run_checkpoint_validation_case(
        tmp_path, monkeypatch, checkpoint, manifest
    )

    assert checked["status"] == "BLOCKED"
    assert "manifest experiment_fingerprint missing or invalid" in checked["errors"]
    assert "checkpoint experiment_fingerprint missing or invalid" in checked["errors"]



def test_prediction_count_rejects_boolean():
    result = {
        "predictions": {
            "y_true": [1],
            "y_prob": [0.8],
            "y_pred": [1],
        },
        "test_counts": {"total": True},
    }
    errors = check_predictions(result)
    assert "test_counts.total missing or invalid" in errors


def test_prediction_labels_reject_booleans():
    result = {
        "predictions": {
            "y_true": [True, 1],
            "y_prob": [0.8, 0.2],
            "y_pred": [1, False],
        }
    }
    errors = check_predictions(result)
    assert any("y_true contains" in item for item in errors)
    assert any("y_pred contains" in item for item in errors)


def test_prediction_sample_ids_must_be_unique():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "y_pred": [0, 1],
            "sample_ids": ["domain-a", "domain-a"],
        }
    }
    errors = check_predictions(result)
    assert "predictions.sample_ids contains duplicate identifiers" in errors


def test_prediction_identity_length_must_match():
    result = {
        "predictions": {
            "y_true": [0, 1],
            "y_prob": [0.2, 0.8],
            "y_pred": [0, 1],
            "test_indices": [10],
        }
    }
    errors = check_predictions(result)
    assert any("test_indices length=1" in item for item in errors)



def test_checkpoint_rejects_index_outside_dataset_bounds(tmp_path, monkeypatch):
    checkpoint, manifest = _valid_checkpoint_and_manifest()
    checkpoint["test_indices"] = [100]
    manifest["test_count"] = 1

    checked = _run_checkpoint_validation_case(
        tmp_path, monkeypatch, checkpoint, manifest, domain_count=100
    )

    assert checked["status"] == "BLOCKED"
    assert any(
        "test_indices contains index outside dataset bounds" in item
        for item in checked["errors"]
    )



def test_baseline_rejects_boolean_or_negative_test_count(tmp_path):
    for label, total in (("boolean", True), ("negative", -1)):
        case_repo = tmp_path / label
        runs = case_repo / "results" / "real" / "runs"
        runs.mkdir(parents=True)
        result = {
            "roc_auc": 0.88,
            "pr_auc": 0.77,
            "f1": 0.74,
            "precision": 0.69,
            "recall": 0.80,
            "test_counts": {"total": total},
        }
        (runs / "baseline_xgboost_tabular_seed_42.json").write_text(
            json.dumps(result), encoding="utf-8"
        )

        checked = validate_baseline_result(case_repo, "tabular", 42)
        assert checked["status"] == "BLOCKED"
        assert "test_counts.total missing or invalid" in checked["errors"]
