# [FIXTURE - meaningless] Provenance, Collection Shortcut, and Infrastructure Connectivity Audit

## 1. Benign Source Disparity (Umbrella vs CESNET)
- **ROC-AUC**: 0.5238
- **Interpretation**: Low/moderate source divergence (AUC <= 0.80)

## 2. Cross-Source Generalization Transfer
- **Primary Malware Feed**: ThreatFox
- **XGBoost Transfer ROC-AUC**: 0.5034

## 3. Feature Sensitivity (Missingness / Null-rate Removal)
- **Full Model AUC**: 0.5102
- **No-Null-Rate AUC**: 0.51
- **AUC Delta**: -0.0002 (Robust to missingness features)

## 4. Graph Infrastructure Reuse Connectivity Audit
| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |
|---|---|---|---|---|
| `resolves_to` | 6.4% | 6.1% | 6.7% | 185 |
| `uses_ns` | 99.6% | 99.6% | 99.5% | 8 |
| `registered_by` | 100.0% | 100.0% | 100.0% | 7 |
| `uses_cert` | 35.1% | 35.4% | 34.8% | 1 |

