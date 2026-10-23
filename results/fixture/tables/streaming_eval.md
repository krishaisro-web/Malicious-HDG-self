# [FIXTURE - meaningless] Streaming Domain Prototype Evaluation

- **Initial Baseline Month ($M_0$)**: `2023-04`
- **Evaluated Months**: `2023-04, 2023-05, 2023-06`
- **Replay Reservoir Capacity**: `1000`
- **Initial $M_0$ Test F1**: `0.6667` | **TPR@0.1% FPR**: `0.0714`

## Continual Learning Comparison ($T \to T+1$)
| Step | Target Month | Strategy | New Month F1 | New Month TPR@0.1% FPR | New Month ROC-AUC | Month 1 F1 | Backward Transfer (ΔF1) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2023-04 | `initial_base` | 0.6667 | 0.0714 | 0.4592 | 0.6667 | **0.0000** |
| 1 | 2023-05 | `retrain_from_scratch` | 0.6400 | 0.0833 | 0.4846 | 0.6667 | **0.0000** |
| 1 | 2023-05 | `naive_finetune` | 0.5538 | 0.0417 | 0.5123 | 0.5714 | **-0.0953** |
| 1 | 2023-05 | `replay` | 0.6486 | 0.0000 | 0.5401 | 0.6667 | **0.0000** |
| 2 | 2023-06 | `retrain_from_scratch` | 0.6739 | 0.0000 | 0.4871 | 0.6667 | **0.0000** |
| 2 | 2023-06 | `naive_finetune` | 0.6593 | 0.0000 | 0.3978 | 0.5600 | **-0.1067** |
| 2 | 2023-06 | `replay` | 0.6739 | 0.0323 | 0.3892 | 0.6667 | **0.0000** |

### Key Findings
1. **Retrain from scratch**: Maintains robust performance across both past and present distributions at the expense of cumulative training time.
2. **Naive fine-tuning**: Adapts rapidly to the active month's malicious patterns but suffers from catastrophic forgetting (negative backward transfer on Month 1).
3. **Replay buffer**: Mitigates catastrophic forgetting by interleaving historical anchor domains, stabilizing long-term recall without full retraining.
