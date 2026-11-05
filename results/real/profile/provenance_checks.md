# Provenance, Collection Shortcut, and Infrastructure Connectivity Audit

## 1. Benign Source Disparity (Umbrella vs CESNET)
- **ROC-AUC**: 0.8416
- **Interpretation**: High source divergence (AUC > 0.80)

## 2. Cross-Source Generalization Transfer
- **Primary Malware Feed**: N/A
- **XGBoost Transfer ROC-AUC**: N/A

## 3. Feature Sensitivity (Missingness / Null-rate Removal)
- **Full Model AUC**: 0.8906
- **No-Null-Rate AUC**: 0.8751
- **AUC Delta**: -0.0155 (Robust to missingness features)

## 4. Graph Infrastructure Reuse Connectivity Audit
| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |
|---|---|---|---|---|
| `resolves_to` | 33.1% | 37.5% | 30.9% | 10157 |
| `uses_ns` | 74.5% | 66.1% | 78.7% | 15645 |
| `registered_by` | 98.8% | 98.9% | 98.8% | 1614 |
| `uses_cert` | 41.2% | 56.7% | 33.5% | 3212 |

