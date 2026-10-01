# [FIXTURE - meaningless] Baseline Model Evaluation (Identical Splits)
*Evaluated across 1 seeded splits (mean ± std)*

| Model | ROC-AUC | F1 Score | PR-AUC |
|---|---|---|---|
| `tfidf_logreg_e2ld` | 0.4761 ± 0.0000 | 0.6708 ± 0.0000 | 0.4897 ± 0.0000 |
| `xgboost_tabular` | 0.4636 ± 0.0000 | 0.6696 ± 0.0000 | 0.4815 ± 0.0000 |
| `xgboost_tabular_lexical` | 0.5008 ± 0.0000 | 0.6708 ± 0.0000 | 0.5014 ± 0.0000 |
| `single_feat_length` | 0.5212 ± 0.0000 | 0.6708 ± 0.0000 | 0.5232 ± 0.0000 |
| `single_feat_no_ip` | 0.4894 ± 0.0000 | 0.6708 ± 0.0000 | 0.4998 ± 0.0000 |
| `mlp_no_edges` | 0.5349 ± 0.0000 | 0.7314 ± 0.0000 | 0.6118 ± 0.0000 |

> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.
