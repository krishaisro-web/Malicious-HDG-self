# [FIXTURE - meaningless] HeteroGNN Evaluation (GROUP Split)
*Evaluated across 1 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.5015 ± 0.0000 | 0.6687 ± 0.0000 | 0.5032 ± 0.0000 | 0.0068 |
| `HeteroGNN (ATTN)` | 0.5345 ± 0.0000 | 0.6687 ± 0.0000 | 0.5361 ± 0.0000 | 0.0000 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 0.0 (p = 1.0)
- **Pooled Paired Bootstrap AUC Difference**: -0.03 (95% CI: [-0.0897, 0.0305], p = 0.24)

