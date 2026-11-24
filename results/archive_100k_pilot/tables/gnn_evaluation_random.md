# HeteroGNN Evaluation (RANDOM Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.9343 ± 0.0031 | 0.8034 ± 0.0075 | 0.8861 ± 0.0064 | 0.1786 |
| `HeteroGNN (ATTN)` | 0.9337 ± 0.0032 | 0.8023 ± 0.0052 | 0.8836 ± 0.0060 | 0.1622 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 1998.0 (p = 0.051526901378260694)
- **Pooled Paired Bootstrap AUC Difference**: 0.0006 (95% CI: [-0.0002, 0.0014], p = 0.182)

### Comparison: `sage_vs_xgboost`
- **Pooled Exact McNemar Statistic (min(b,c))**: 2786.0 (p = 0.0)
- **Pooled Paired Bootstrap AUC Difference**: 0.0496 (95% CI: [0.0478, 0.0515], p = 0.0)

