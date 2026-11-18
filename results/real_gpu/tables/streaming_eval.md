# Streaming Domain Prototype Evaluation

> [!WARNING]
> **Status: BLOCKED_INSUFFICIENT_DATA**
> Need at least 2 valid months with both classes for continual learning, but found 1.

### Monthly Class Distribution in Dataset
| Month | Total Records | Malware Count | Benign Count | Status |
|:---:|:---:|:---:|:---:|:---:|
| `2022-12` | 7,776 | 0 | 7,776 | Single-Class Only |
| `2023-02` | 7,478 | 0 | 7,478 | Single-Class Only |
| `2023-07` | 13,539 | 13,539 | 0 | Single-Class Only |
| `2023-11` | 9,407 | 9,407 | 0 | Single-Class Only |
| `2023-12` | 53,943 | 2,448 | 51,495 | Eligible (Mixed) |
| `2024-01` | 38 | 38 | 0 | Single-Class Only |
| `2024-02` | 713 | 713 | 0 | Single-Class Only |
| `2024-03` | 7,106 | 7,106 | 0 | Single-Class Only |

> [!NOTE]
> Multi-month continual learning evaluation requires multiple transitions between consecutive months containing both classes.
> To evaluate temporal robustness on datasets with a single overlapping month, refer to the Single Chronological Holdout results in `gnn_evaluation_time.json`.
