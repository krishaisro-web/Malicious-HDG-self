# HeteroGNN Evaluation (GROUP Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.7929 ± 0.0346 | 0.5789 ± 0.0254 | 0.6418 ± 0.0248 | 0.0001 |
| `HeteroGNN (ATTN)` | 0.7437 ± 0.1092 | 0.5322 ± 0.1022 | 0.5361 ± 0.1099 | 0.0000 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 11846.0 (p = 0.19794012207185813)
- **Pooled Paired Bootstrap AUC Difference**: 0.046 (95% CI: [0.0418, 0.0499], p = 0.0)

