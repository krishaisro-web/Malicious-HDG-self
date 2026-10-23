# [FIXTURE - meaningless] HeteroGNN Evaluation (RANDOM Split)
*Evaluated across 1 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.4689 ± 0.0000 | 0.6682 ± 0.0000 | 0.4912 ± 0.0000 | 0.0000 |
| `HeteroGNN (ATTN)` | 0.4902 ± 0.0000 | 0.6682 ± 0.0000 | 0.4997 ± 0.0000 | 0.0000 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 0.0 (p = 1.0)
- **Pooled Paired Bootstrap AUC Difference**: -0.0217 (95% CI: [-0.0736, 0.0297], p = 0.52)

