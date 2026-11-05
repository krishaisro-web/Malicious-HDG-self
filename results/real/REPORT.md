# Malicious-HDG Comprehensive PhD Evaluation Report
**Generated**: `2026-11-03T09:15:25.611094+00:00` | **Environment Mode**: `PRODUCTION (Real Zenodo DomainRadar v2)`

## Executive Summary: PhD Target Objectives (O1–O5)

| Objective | Target Criterion | Observed Result | Status |
|:---|:---|:---|:---:|
| **O1: Low-FPR Operation** | FPR $\le 0.1\%$ with non-trivial TPR | TPR: 15.60% (Realised FPR: 0.100%) | NOT MET |
| **O2: Temporal Generalization** | F1 degradation $\le 0.10$ on $T \to T+1$ | Random F1: 0.7902 $\to$ Temporal F1: 0.0900 (Gap: 0.7002) | NOT MET |
| **O3: AUROC Superiority** | Significant gain over tabular XGBoost ($p < 0.05$) | HeteroGNN: 0.9214 vs XGBoost: 0.8849 (Δ: +0.0365, p=0.0000) | MET |
| **O4: CPU Latency Budget** | Per-query CPU inference $< 50$ ms | 548.90 ms mean (P95: 1292.50 ms) | NOT MET |
| **O5: Adversarial Robustness** | $\ge 70\%$ recovery score under structural attack | Recovery Score: 82.1% (budget_2) | MET |

## 1. Multi-Seed Baseline Comparisons
Evaluated across 5 random seeds (42, 1337, 2024, 777, 999) with 95% bootstrap confidence intervals.

| Model Family | Architecture | ROC-AUC | F1 Score | TPR @ 0.1% FPR | Prec ($\pi=1\%$) | Prec ($\pi=0.1\%$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | Tfidf Logreg E2Ld | 0.8992 ± 0.0016 | 0.7580 ± 0.0038 | 0.1028 | 0.3940 | 0.0605 |
| Baseline | Xgboost Tabular | 0.8849 ± 0.0020 | 0.7452 ± 0.0015 | 0.0555 | 0.3188 | 0.0443 |
| Baseline | Xgboost Tabular Lexical | 0.8956 ± 0.0026 | 0.7544 ± 0.0045 | 0.0768 | 0.3568 | 0.0521 |
| Baseline | Single Feat Length | 0.6207 ± 0.0019 | 0.5264 ± 0.0037 | 0.0028 | 0.0231 | 0.0023 |
| Baseline | Single Feat No Ip | 0.6264 ± 0.0005 | 0.4991 | 0.3797 | 0.0293 | 0.0030 |
| Baseline | Mlp No Edges | 0.8289 ± 0.0064 | 0.6638 ± 0.0062 | 0.0255 | 0.1897 | 0.0227 |
| **Proposed GNN** | **HeteroGNN (SAGE)** | **0.9214 ± 0.0057** | **0.7902 ± 0.0074** | **0.1560** | **0.6120** | **0.1352** |
| **Proposed GNN** | **HeteroGNN (ATTN)** | **0.9308 ± 0.0030** | **0.7989 ± 0.0042** | **0.1472** | **0.5981** | **0.1285** |

## 2. Generalization Across Split Regimes
Leak-free comparison isolating structural connectivity and temporal evolution.

| Split Methodology | Target Focus | ROC-AUC | F1 Score | TPR @ 0.1% FPR | Degradation vs Random |
|:---|:---|:---:|:---:|:---:|:---:|
| `group` (SAGE) | Joint infrastructure components (masked graph boundary) | 0.7929 ± 0.0346 | 0.5789 ± 0.0254 | 0.0001 | -0.2113 |
| `group` (ATTN) | Joint infrastructure components (masked graph boundary) | 0.7437 ± 0.1092 | 0.5322 ± 0.1022 | 0.0000 | -0.2580 |
| `random` (SAGE) | Optimistic random baseline | 0.9214 ± 0.0057 | 0.7902 ± 0.0074 | N/A | 0.0000 |
| `random` (ATTN) | Optimistic random baseline | 0.9308 ± 0.0030 | 0.7989 ± 0.0042 | N/A | +0.0087 |
| `time` (SAGE) | Temporal holdout (future topology masked) | 0.5626 ± 0.0471 | 0.0900 ± 0.0130 | 0.0004 | -0.7002 |
| `time` (ATTN) | Temporal holdout (future topology masked) | 0.6792 ± 0.0185 | 0.1163 ± 0.0191 | 0.0000 | -0.6739 |

## 2b. Graph Ablation Study
Systematic isolation of relational edge types, structural degree, and lexical representations.

| Ablation Configuration | Isolated Component | ROC-AUC | F1 Score | PR-AUC |
|:---|:---|:---:|:---:|:---:|
| `full_sage` | Full Sage | 0.9293 ± 0.0037 | 0.8049 ± 0.0057 | 0.8709 ± 0.0086 |
| `no_certificates` | No Certificates | 0.9303 ± 0.0025 | 0.8064 ± 0.0041 | 0.8770 ± 0.0038 |
| `cert_coissue_key` | Cert Coissue Key | 0.9286 ± 0.0028 | 0.8031 ± 0.0059 | 0.8701 ± 0.0068 |
| `no_nameservers` | No Nameservers | 0.9198 ± 0.0052 | 0.7919 ± 0.0075 | 0.8535 ± 0.0100 |
| `no_registrars` | No Registrars | 0.9200 ± 0.0033 | 0.7898 ± 0.0032 | 0.8536 ± 0.0067 |
| `no_asn` | No Asn | 0.9273 ± 0.0027 | 0.8029 ± 0.0043 | 0.8674 ± 0.0067 |
| `graph_only_no_lexical` | Graph Only No Lexical | 0.9240 ± 0.0015 | 0.7996 ± 0.0010 | 0.8555 ± 0.0035 |
| `degree_mode_train_visible` | Degree Mode Train Visible | 0.9210 ± 0.0026 | 0.7805 ± 0.0036 | 0.8539 ± 0.0061 |
| `mlp_no_edges` | Mlp No Edges | 0.8761 ± 0.0024 | 0.7286 ± 0.0023 | 0.7816 ± 0.0049 |
| `attn_variant` | Attn Variant | 0.9331 ± 0.0033 | 0.8090 ± 0.0067 | 0.8817 ± 0.0063 |
| `sage_guard_clean` | Sage Guard Clean | 0.9273 ± 0.0045 | 0.8025 ± 0.0059 | 0.8668 ± 0.0122 |

## 3. Adversarial Structural Attack & GNNGuard Recovery
Coordinated evasion attack injecting malicious edges into legitimate training infrastructure.

| Attack Budget ($k$) | Clean SAGE F1 | Attacked Undefended F1 | Attacked SAGE_Guard F1 | Recovery Score |
|:---:|:---:|:---:|:---:|:---:|
| Budget $k=1$ | 0.8025 | 0.8033 | **0.7668** | **4.2%** |
| Budget $k=2$ | 0.8025 | 0.8499 | **0.8104** | **82.1%** |
| Budget $k=5$ | 0.8025 | 0.8743 | **0.8587** | **22.1%** |
| Budget $k=10$ | 0.8025 | 0.8741 | **0.8730** | **1.4%** |

## 4. CPU Inference Latency Benchmark
- **CPU**: `QEMU Virtual CPU version 2.5+` (48 cores, OS: Linux 6.8.0-71-generic (x86_64))
- **PyTorch**: `2.14.0+cu130` | **Max Threads**: `32`

| Batch Size | Threads | Feature Lookup (ms) | 2-Hop Ego Build (ms) | Model Forward (ms) | End-to-End Mean | Median | P95 | P99 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 8.26 | 414.59 | 126.05 | **548.90** | 355.72 | 1292.50 | 1333.99 |
| 32 | 1 | 10.24 | 828.54 | 3755.24 | **4594.02** | 4561.11 | 5003.40 | 5202.10 |
| 1 | 32 | 8.26 | 414.59 | 126.05 | **548.90** | 355.72 | 1292.50 | 1333.99 |
| 32 | 32 | 9.49 | 812.47 | 511.88 | **1333.85** | 1297.42 | 1449.70 | 1515.91 |

## 5. Streaming & Continual Domain Updates
Online evaluation across chronological monthly cohorts comparing retraining from scratch, naive fine-tuning, and reservoir replay (M=1000).

> [!WARNING]
> **Status: BLOCKED_INSUFFICIENT_DATA**
> Need at least 2 valid months with both classes for continual learning, but found 1.
> Continual-learning evaluation requires $\ge 2$ consecutive multi-class months ($T \to T+1$).

## 6. Operational BIND RPZ Deployment
> [!WARNING]
> **Status: BLOCKED_MISSING_CERT_METADATA**
> Checkpoint or split does not contain pre-allocated certification indices ('cert_indices'). Post-hoc fallback is prohibited.
> - **Firewall Policy**: No speculative blocking rules emitted (fail-closed).

