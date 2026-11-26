# HeteroGNN Evaluation (GROUP Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.7798 ± 0.0432 | 0.4028 ± 0.0181 | 0.4761 ± 0.0429 | 0.0006 |
| `HeteroGNN (ATTN)` | 0.8375 ± 0.0103 | 0.5235 ± 0.0232 | 0.5345 ± 0.0225 | 0.0038 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 37147.0 (p = 0.0)
- **Pooled Paired Bootstrap AUC Difference**: -0.0832 (95% CI: [-0.0849, -0.0814], p = 0.0)

