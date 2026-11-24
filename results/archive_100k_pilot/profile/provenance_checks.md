# Provenance, Collection Shortcut, and Infrastructure Connectivity Audit

## 1. Benign Source Disparity (Umbrella vs CESNET)
- **ROC-AUC**: 0.8404
- **Interpretation**: High source divergence (AUC > 0.80)

## 2. Cross-Source Generalization Transfer
- **Primary Malware Feed**: ThreatFox
- **XGBoost Transfer ROC-AUC**: 0.6891
- **TF-IDF Transfer ROC-AUC**: 0.4732
- **HeteroGNN Transfer ROC-AUC**: 0.7074

## 3. Feature Sensitivity (Missingness / Null-rate Removal)
- **Full Model AUC**: 0.8905
- **No-Null-Rate AUC**: 0.8741
- **AUC Delta**: -0.0164 (Robust to missingness features)

## 4. Graph Infrastructure Reuse Connectivity Audit
| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |
|---|---|---|---|---|
| `resolves_to` | 33.1% | 37.5% | 30.9% | 10157 |
| `uses_ns` | 74.5% | 66.1% | 78.7% | 15645 |
| `registered_by` | 65.7% | 68.3% | 64.4% | 1613 |
| `uses_cert` | 15.9% | 10.5% | 18.6% | 3211 |

