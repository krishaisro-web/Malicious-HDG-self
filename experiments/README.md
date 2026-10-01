# Auxiliary & Historical Experiments

This directory contains standalone exploratory experiments and secondary comparative baselines outside the primary Zenodo DomainRadar v2 pipeline.

## Subdirectories

### `chrmor_lexical/`
Contains scripts evaluating lexical and DGA baseline classifiers on the Chrmor DGA benchmark:
- `01_preprocess_chrmor.py`: Cleans raw Chrmor CSV data, removes conflicting/empty domains, and standardizes labels.
- `02_chrmor_lexical_baseline.py`: Evaluates a TF-IDF + Logistic Regression lexical classifier with family-level breakdown and ROC-AUC / F1 metrics.
