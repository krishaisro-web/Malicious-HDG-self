# Legacy v1 Architecture (Archived)

## Notice: FROZEN ARCHIVE

This directory contains the legacy (v1) exploratory code, models, checkpoints, test scripts, and initial results created during earlier iterations of the Malicious-HDG project.

**Status: FROZEN / READ-ONLY**
- **Do NOT import from this directory.** The active and maintained package is `src.hdg`.
- **Do NOT run experiments from this directory.** Active experiments are driven by `pipeline/`.
- **Excluded from test collection.** Pytest excludes `archive/` via `norecursedirs` in `pyproject.toml`.

## Directory Contents

- `scripts/`: Initial exploratory pipeline scripts (`00_check_dataset.py` through `18_static_baseline_check.py`).
- `models/`: Obsolete initial models (`classifier.py`, `encoder.py`, `full_model.py`, `temporal.py`, and `hdg_legacy/`), which relied on a 2-layer snapshot model without GNNGuard or full multi-relation message passing.
- `checkpoints/`: Initial v1 checkpoint `.pt` files (`best_model*.pt`, `last_model.pt`). Active checkpoints are stored under `artifacts/checkpoints/{fixture,real}/`.
- `results/`: Exploratory evaluation results, confusion matrices, and early training logs. Active results are under `results/{fixture,real,chrmor}/`.
- `tests/`: Tests covering only legacy components (`test_full_model.py`). Active tests are maintained under `tests/`.
