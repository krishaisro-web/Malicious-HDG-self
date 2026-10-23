# [FIXTURE - meaningless] Baseline Model Evaluation (Identical Splits)
*Evaluated across 1 seeded splits (mean ± std)*

| Model | ROC-AUC | F1 Score | PR-AUC |
|---|---|---|---|
| `tfidf_logreg_e2ld` | 0.5061 ± 0.0000 | 0.6682 ± 0.0000 | 0.5128 ± 0.0000 |
| `xgboost_tabular` | 0.4864 ± 0.0000 | 0.6682 ± 0.0000 | 0.4928 ± 0.0000 |
| `xgboost_tabular_lexical` | 0.4977 ± 0.0000 | 0.6682 ± 0.0000 | 0.4953 ± 0.0000 |
| `single_feat_length` | 0.4642 ± 0.0000 | 0.6682 ± 0.0000 | 0.4797 ± 0.0000 |
| `single_feat_no_ip` | 0.4984 ± 0.0000 | 0.6682 ± 0.0000 | 0.5009 ± 0.0000 |
| `mlp_no_edges` | 0.4993 ± 0.0000 | 0.6682 ± 0.0000 | 0.5124 ± 0.0000 |

> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.
