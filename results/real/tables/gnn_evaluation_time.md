# HeteroGNN Evaluation (TIME Split)
*Evaluated across 5 seeded runs (mean ± std)*

| Architecture | ROC-AUC | F1 Score | PR-AUC | TPR @ 0.1% FPR |
|---|---|---|---|---|
| `HeteroGNN (SAGE)` | 0.5626 ± 0.0471 | 0.0900 ± 0.0130 | 0.0508 ± 0.0068 | 0.0004 |
| `HeteroGNN (ATTN)` | 0.6792 ± 0.0185 | 0.1163 ± 0.0191 | 0.0953 ± 0.0063 | 0.0000 |

## Paired Hypothesis Testing (Across All Seeds)
### Comparison: `sage_vs_attn`
- **Pooled Exact McNemar Statistic (min(b,c))**: 57427.0 (p = 1.5598558775116499e-62)
- **Pooled Paired Bootstrap AUC Difference**: -0.1276 (95% CI: [-0.1347, -0.1212], p = 0.0)

