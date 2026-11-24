# Baseline Model Evaluation (Identical Splits)
*Evaluated across 5 seeded splits (mean ± std)*

| Model | ROC-AUC | F1 Score | PR-AUC |
|---|---|---|---|
| `tfidf_logreg_e2ld` | 0.9053 ± 0.0007 | 0.7574 ± 0.0041 | 0.8257 ± 0.0016 |
| `xgboost_tabular` | 0.8848 ± 0.0020 | 0.7268 ± 0.0024 | 0.7731 ± 0.0026 |
| `xgboost_tabular_lexical` | 0.8951 ± 0.0026 | 0.7411 ± 0.0058 | 0.8085 ± 0.0053 |
| `single_feat_length` | 0.6207 ± 0.0019 | 0.5232 ± 0.0039 | 0.4147 ± 0.0028 |
| `single_feat_no_ip` | 0.6264 ± 0.0005 | 0.4991 ± 0.0000 | 0.4329 ± 0.0007 |
| `mlp_no_edges` | 0.8301 ± 0.0021 | 0.6538 ± 0.0052 | 0.7021 ± 0.0026 |

> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.
