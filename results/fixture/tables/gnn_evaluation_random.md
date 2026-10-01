# [FIXTURE - meaningless] HeteroGNN Evaluation (RANDOM Split)
*Evaluated across 1 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.4903 ± 0.0000 | 0.7282 ± 0.0000 | 0.5749 ± 0.0000 | 0.0000 |
| `HeteroGNN (ATTN)` | 0.5203 ± 0.0000 | 0.7314 ± 0.0000 | 0.5941 ± 0.0000 | 0.0092 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 0.0 (p = 0.25)
- **Pooled Paired Bootstrap AUC Difference**: -0.0326 (95% CI: [-0.1151, 0.0403], p = 0.44)

