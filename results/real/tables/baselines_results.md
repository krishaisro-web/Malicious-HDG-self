# Baseline Model Evaluation (Identical Splits)
*Evaluated across 5 seeded splits (mean ± std)*

| Model | ROC-AUC | F1 Score | PR-AUC |
|---|---|---|---|
| `tfidf_logreg_e2ld` | 0.8992 ± 0.0016 | 0.7580 ± 0.0038 | 0.8144 ± 0.0032 |
| `xgboost_tabular` | 0.8849 ± 0.0020 | 0.7452 ± 0.0015 | 0.7732 ± 0.0027 |
| `xgboost_tabular_lexical` | 0.8956 ± 0.0026 | 0.7544 ± 0.0045 | 0.8092 ± 0.0053 |
| `single_feat_length` | 0.6207 ± 0.0019 | 0.5264 ± 0.0037 | 0.4147 ± 0.0028 |
| `single_feat_no_ip` | 0.6264 ± 0.0005 | 0.4991 ± 0.0000 | 0.4329 ± 0.0007 |
| `mlp_no_edges` | 0.8289 ± 0.0064 | 0.6638 ± 0.0062 | 0.7009 ± 0.0105 |

> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.
