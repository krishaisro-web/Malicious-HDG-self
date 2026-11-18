# HeteroGNN Evaluation (TIME Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.6226 ± 0.0116 | 0.1096 ± 0.0040 | 0.0597 ± 0.0020 | 0.0324 |
| `HeteroGNN (ATTN)` | 0.6153 ± 0.0245 | 0.1065 ± 0.0087 | 0.0614 ± 0.0051 | 0.0671 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 26342.0 (p = 3.947670052701871e-133)
- **Pooled Paired Bootstrap AUC Difference**: 0.0112 (95% CI: [0.0079, 0.0143], p = 0.0)

