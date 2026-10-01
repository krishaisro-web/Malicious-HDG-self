# [FIXTURE - meaningless] CPU Inference Latency Benchmark

## System Environment
- **CPU Model**: `Intel64 Family 6 Model 165 Stepping 2, GenuineIntel`
- **Physical/Logical Cores**: `12`
- **Operating System**: `Windows 11 (AMD64)`
- **PyTorch Version**: `2.14.0+cpu`
- **Queries Measured**: `50` (Warmup: `5`)

## Benchmark Results (ms per query)
| Batch Size | Threads | Feature Lookup Mean (P95) | 2-Hop Ego Build Mean (P95) | Model Forward Mean (P95) | End-to-End Mean | Median | P95 | P99 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 1.37 (1.97) | 16.12 (21.05) | 30.77 (40.60) | **48.25** | 45.31 | 64.95 | 88.48 |
| 32 | 1 | 1.36 (1.72) | 31.44 (35.21) | 40.80 (51.69) | **73.60** | 71.77 | 89.27 | 90.53 |
| 1 | 12 | 1.77 (2.10) | 25.05 (33.00) | 25.17 (34.08) | **51.99** | 51.62 | 68.30 | 71.27 |
| 32 | 12 | 1.97 (2.43) | 56.70 (64.77) | 34.73 (48.03) | **93.40** | 90.51 | 117.95 | 123.46 |

> Detailed methodology and ego-subgraph mathematical equivalence documented in `docs/latency.md`.
