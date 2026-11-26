# HeteroGNN Evaluation (TIME Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.6217 ± 0.0077 | 0.0465 ± 0.0032 | 0.0243 ± 0.0003 | 0.0358 |
| `HeteroGNN (ATTN)` | 0.6403 ± 0.0185 | 0.0506 ± 0.0034 | 0.0261 ± 0.0020 | 0.0513 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 122818.0 (p = 0.0)
- **Pooled Paired Bootstrap AUC Difference**: -0.0213 (95% CI: [-0.0231, -0.0194], p = 0.0)

