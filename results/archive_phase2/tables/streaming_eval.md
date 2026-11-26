# Streaming Domain Prototype Evaluation

> [!WARNING]
> **Status: BLOCKED_INSUFFICIENT_DATA**
> Need at least 2 valid months with both classes for continual learning, but found 1.

### Monthly Class Distribution in Dataset
| Month | Total Records | Malware Count | Benign Count | Status |
|:---:|:---:|:---:|:---:|:---:|
| `2022-12` | 49,022 | 0 | 49,022 | Single-Class Only |
| `2023-02` | 46,439 | 0 | 46,439 | Single-Class Only |
| `2023-07` | 34,141 | 34,141 | 0 | Single-Class Only |
| `2023-11` | 23,704 | 23,704 | 0 | Single-Class Only |
| `2023-12` | 330,111 | 6,006 | 324,105 | Eligible (Mixed) |
| `2024-01` | 88 | 88 | 0 | Single-Class Only |
| `2024-02` | 1,844 | 1,844 | 0 | Single-Class Only |
| `2024-03` | 17,571 | 17,571 | 0 | Single-Class Only |

> [!NOTE]
> Multi-month continual learning evaluation requires multiple transitions between consecutive months containing both classes.
> To evaluate temporal robustness on datasets with a single overlapping month, refer to the Single Chronological Holdout results in `gnn_evaluation_time.json`.
