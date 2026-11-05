# CPU Inference Latency Benchmark

## System Environment
- **CPU Model**: `QEMU Virtual CPU version 2.5+`
- **Physical/Logical Cores**: `48`
- **Operating System**: `Linux 6.8.0-71-generic (x86_64)`
- **PyTorch Version**: `2.14.0+cu130`
- **Queries Measured**: `500` (Warmup: `20`)

## Benchmark Results (ms per query)
| Batch Size | Threads | Feature Lookup Mean (P95) | 2-Hop Ego Build Mean (P95) | Model Forward Mean (P95) | End-to-End Mean | Median | P95 | P99 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 8.25 (8.91) | 248.18 (480.91) | 983.44 (2509.27) | **1239.86** | 565.53 | 3003.98 | 3044.28 |
| 32 | 1 | 10.24 (11.51) | 828.54 (941.02) | 3755.24 (4101.52) | **4594.02** | 4561.11 | 5003.40 | 5202.10 |
| 1 | 32 | 8.29 (8.81) | 243.73 (463.84) | 156.92 (318.06) | **408.94** | 322.14 | 793.38 | 813.57 |
| 32 | 32 | 9.49 (10.33) | 812.47 (927.56) | 511.88 (538.16) | **1333.85** | 1297.42 | 1449.70 | 1515.91 |

> Detailed methodology and ego-subgraph mathematical equivalence documented in `docs/latency.md`.
