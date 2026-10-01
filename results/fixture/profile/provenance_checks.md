# [FIXTURE - meaningless] Provenance, Collection Shortcut, and Infrastructure Connectivity Audit

## 1. Benign Source Disparity (Umbrella vs CESNET)
- **ROC-AUC**: 0.5195
- **Interpretation**: Low/moderate source divergence (AUC <= 0.80)

## 2. Cross-Source Generalization Transfer
- **Primary Malware Feed**: ThreatFox
- **XGBoost Transfer ROC-AUC**: 0.4976

## 3. Feature Sensitivity (Missingness / Null-rate Removal)
- **Full Model AUC**: 0.5034
- **No-Null-Rate AUC**: 0.502
- **AUC Delta**: -0.0013 (Robust to missingness features)

## 4. Graph Infrastructure Reuse Connectivity Audit
| Relation | Overall Share Degree >= 2 | Malware Share Degree >= 2 | Benign Share Degree >= 2 | Shared Nodes |
|---|---|---|---|---|
| `resolves_to` | 7.5% | 6.6% | 6.7% | 185 |
| `uses_ns` | 116.6% | 99.6% | 99.5% | 8 |
| `registered_by` | 117.1% | 100.0% | 100.0% | 7 |
| `uses_cert` | 41.1% | 35.0% | 35.0% | 1 |

