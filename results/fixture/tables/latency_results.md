# [FIXTURE - meaningless] CPU Inference Latency Benchmark

## System Environment
- **CPU Model**: `arm`
- **Physical/Logical Cores**: `10`
- **Operating System**: `Darwin 27.0.0 (arm64)`
- **PyTorch Version**: `2.14.1`
- **Queries Measured**: `20` (Warmup: `5`)

## Benchmark Results (ms per query)
| Batch Size | Threads | Feature Lookup Mean (P95) | 2-Hop Ego Build Mean (P95) | Model Forward Mean (P95) | End-to-End Mean | Median | P95 | P99 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 0.21 (0.27) | 4.44 (5.42) | 3.33 (3.88) | **7.97** | 8.20 | 9.38 | 9.73 |
| 32 | 1 | 0.24 (0.28) | 8.46 (8.56) | 4.93 (5.18) | **13.64** | 13.58 | 13.95 | 13.98 |
| 1 | 2 | 0.19 (0.23) | 4.38 (5.36) | 4.62 (5.13) | **9.19** | 9.53 | 10.66 | 10.75 |
| 32 | 2 | 0.24 (0.25) | 8.48 (8.58) | 6.11 (6.38) | **14.84** | 14.87 | 15.04 | 15.07 |

> Detailed methodology and ego-subgraph mathematical equivalence documented in `docs/latency.md`.
