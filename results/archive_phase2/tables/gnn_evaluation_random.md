# HeteroGNN Evaluation (RANDOM Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.9307 ± 0.0016 | 0.6952 ± 0.0028 | 0.7716 ± 0.0060 | 0.1740 |
| `HeteroGNN (ATTN)` | 0.9291 ± 0.0041 | 0.6967 ± 0.0050 | 0.7692 ± 0.0102 | 0.1603 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 7118.0 (p = 7.349371775102301e-12)
- **Pooled Paired Bootstrap AUC Difference**: 0.0015 (95% CI: [0.001, 0.002], p = 0.0)

### Comparison: `sage_vs_xgboost`
- **Pooled Exact McNemar Statistic (min(b,c))**: 12111.0 (p = 0.0)
- **Pooled Paired Bootstrap AUC Difference**: 0.0454 (95% CI: [0.0444, 0.0463], p = 0.0)

