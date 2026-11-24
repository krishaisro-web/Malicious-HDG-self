# HeteroGNN Evaluation (GROUP Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.7994 ± 0.0146 | 0.5632 ± 0.0212 | 0.6571 ± 0.0143 | 0.0004 |
| `HeteroGNN (ATTN)` | 0.8077 ± 0.0061 | 0.6057 ± 0.0147 | 0.6610 ± 0.0084 | 0.0002 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 10182.0 (p = 0.018131505106001145)
- **Pooled Paired Bootstrap AUC Difference**: -0.0163 (95% CI: [-0.0195, -0.0133], p = 0.0)

