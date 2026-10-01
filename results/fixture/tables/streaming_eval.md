# [FIXTURE - meaningless] Streaming Domain Prototype Evaluation

- **Initial Baseline Month ($M_0$)**: `2023-04`
- **Evaluated Months**: `2023-04, 2023-05, 2023-06, 2023-07`
- **Replay Reservoir Capacity**: `1000`
- **Initial $M_0$ Test F1**: `0.6234` | **TPR@0.1% FPR**: `0.0000`

## Continual Learning Comparison ($T \to T+1$)
| Step | Target Month | Strategy | New Month F1 | New Month TPR@0.1% FPR | New Month ROC-AUC | Month 1 F1 | Backward Transfer (ΔF1) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2023-04 | `initial_base` | 0.6234 | 0.0000 | 0.4388 | 0.6234 | **0.0000** |
| 1 | 2023-05 | `retrain_from_scratch` | 0.6087 | 0.0000 | 0.5031 | 0.6133 | **-0.0101** |
| 1 | 2023-05 | `naive_finetune` | 0.5938 | 0.0000 | 0.4969 | 0.6571 | **+0.0337** |
| 1 | 2023-05 | `replay` | 0.5714 | 0.0000 | 0.4676 | 0.5915 | **-0.0319** |
| 2 | 2023-06 | `retrain_from_scratch` | 0.6747 | 0.0000 | 0.5172 | 0.6571 | **+0.0337** |
| 2 | 2023-06 | `naive_finetune` | 0.6739 | 0.0000 | 0.4710 | 0.6667 | **+0.0433** |
| 2 | 2023-06 | `replay` | 0.6739 | 0.0000 | 0.5108 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `retrain_from_scratch` | 0.6667 | 0.0667 | 0.5089 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `naive_finetune` | 0.6667 | 0.1000 | 0.5511 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `replay` | 0.6667 | 0.0333 | 0.5300 | 0.6667 | **+0.0433** |

### Key Findings
1. **Retrain from scratch**: Maintains robust performance across both past and present distributions at the expense of cumulative training time.
2. **Naive fine-tuning**: Adapts rapidly to the active month's malicious patterns but suffers from catastrophic forgetting (negative backward transfer on Month 1).
3. **Replay buffer**: Mitigates catastrophic forgetting by interleaving historical anchor domains, stabilizing long-term recall without full retraining.
