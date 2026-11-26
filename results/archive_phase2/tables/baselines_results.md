# Baseline Model Evaluation (Identical Splits)
*Evaluated across 5 seeded splits (mean ± std)*

| Model | ROC-AUC | F1 Score | PR-AUC |
|---|---|---|---|
| `tfidf_logreg_e2ld` | 0.9151 ± 0.0018 | 0.6409 ± 0.0029 | 0.7250 ± 0.0040 |
| `xgboost_tabular` | 0.8853 ± 0.0016 | 0.5971 ± 0.0029 | 0.6087 ± 0.0049 |
| `xgboost_tabular_lexical` | 0.8961 ± 0.0016 | 0.6141 ± 0.0048 | 0.6711 ± 0.0055 |
| `single_feat_length` | 0.6214 ± 0.0026 | 0.3207 ± 0.0005 | 0.2229 ± 0.0017 |
| `single_feat_no_ip` | 0.6284 ± 0.0028 | 0.3793 ± 0.0046 | 0.2459 ± 0.0027 |
| `mlp_no_edges` | 0.8154 ± 0.0075 | 0.4861 ± 0.0100 | 0.4823 ± 0.0108 |

> **Note**: All metrics represent multi-seed evaluations. Never report single best seed.
