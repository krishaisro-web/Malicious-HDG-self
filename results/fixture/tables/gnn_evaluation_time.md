# [FIXTURE - meaningless] HeteroGNN Evaluation (TIME Split)
*Evaluated across 1 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.6183 ± 0.0000 | 0.6000 ± 0.0000 | 0.6104 ± 0.0000 | 0.0861 |
| `HeteroGNN (ATTN)` | 0.6184 ± 0.0000 | 0.6318 ± 0.0000 | 0.5751 ± 0.0000 | 0.0927 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 34.0 (p = 0.04221061162240417)
- **Pooled Paired Bootstrap AUC Difference**: -0.0038 (95% CI: [-0.0585, 0.0487], p = 0.92)

