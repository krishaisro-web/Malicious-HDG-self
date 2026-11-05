# HeteroGNN Evaluation (RANDOM Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.9214 ± 0.0057 | 0.7902 ± 0.0074 | 0.8562 ± 0.0142 | 0.1029 |
| `HeteroGNN (ATTN)` | 0.9308 ± 0.0030 | 0.7989 ± 0.0042 | 0.8790 ± 0.0068 | 0.1674 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 2240.0 (p = 5.269815890786802e-09)
- **Pooled Paired Bootstrap AUC Difference**: -0.0093 (95% CI: [-0.0102, -0.0084], p = 0.0)

