# Provenance, Collection Shortcut, and Infrastructure Connectivity Audit

## 1. Benign Source Disparity (Umbrella vs CESNET)
- **ROC-AUC**: 0.8473
- **Interpretation**: High source divergence (AUC > 0.80)

## 2. Cross-Source Generalization Transfer
- **Primary Malware Feed**: ThreatFox
- **XGBoost Transfer ROC-AUC**: 0.7014
- **TF-IDF Transfer ROC-AUC**: 0.5209
- **HeteroGNN Transfer ROC-AUC**: 0.6992

## 3. Feature Sensitivity (Missingness / Null-rate Removal)
- **Full Model AUC**: 0.8897
- **No-Null-Rate AUC**: 0.874
- **AUC Delta**: -0.0157 (Robust to missingness features)

## 4. Graph Infrastructure Reuse Connectivity Audit
| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |
|---|---|---|---|---|
| `resolves_to` | 45.7% | 46.3% | 45.6% | 80921 |
| `uses_ns` | 81.7% | 69.1% | 84.2% | 49888 |
| `registered_by` | 65.5% | 68.7% | 64.9% | 3147 |
| `uses_cert` | 23.4% | 12.4% | 25.6% | 17938 |

