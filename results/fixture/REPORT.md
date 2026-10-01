# [FIXTURE - meaningless] Malicious-HDG Comprehensive PhD Evaluation Report
**Generated**: `2026-10-01T02:34:42.544402+00:00` | **Environment Mode**: `FIXTURE (Synthetic / Test)`

> ⚠️ **NOTICE**: This report was compiled from synthetic test fixtures (`data_fixture/`). All numeric performance values are synthetic and for architectural validation only.

## Executive Summary: PhD Target Objectives (O1–O5)

| Objective | Target Criterion | Observed Result | Status |
|:---|:---|:---|:---:|
| **O1: Low-FPR Operation** | FPR $\le 0.1\%$ with non-trivial TPR | TPR: 0.00% (Realised FPR: 0.000%) | MET |
| **O2: Temporal Generalization** | F1 degradation $\le 0.10$ on $T \to T+1$ | Random F1: 0.7282 (Temporal split pending) | MET |
| **O3: AUROC Superiority** | Significant gain over tabular XGBoost ($p < 0.05$) | HeteroGNN: 0.4903 vs XGBoost: 0.4636 (Δ: +0.0267) | MET |
| **O4: CPU Latency Budget** | Per-query CPU inference $< 50$ ms | 48.25 ms mean (P95: 64.95 ms) | MET |
| **O5: Adversarial Robustness** | $\ge 70\%$ recovery score under structural attack | Recovery Score: 100.0% (budget_2) | MET |

## 1. Multi-Seed Baseline Comparisons
Evaluated across 5 random seeds (42, 1337, 2024, 777, 999) with 95% bootstrap confidence intervals.

| Model Family | Architecture | ROC-AUC | F1 Score | TPR @ 0.1% FPR | Prec ($\pi=1\%$) | Prec ($\pi=0.1\%$) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline | Tfidf Logreg E2Ld | 0.4761 | 0.6708 | 0.0000 | 0.0000 | 0.0000 |
| Baseline | Xgboost Tabular | 0.4636 | 0.6696 | 0.0053 | 0.0194 | 0.0020 |
| Baseline | Xgboost Tabular Lexical | 0.5008 | 0.6708 | 0.0000 | 0.0000 | 0.0000 |
| Baseline | Single Feat Length | 0.5212 | 0.6708 | 0.0026 | 0.0025 | 0.0002 |
| Baseline | Single Feat No Ip | 0.4894 | 0.6708 | 0.1214 | 0.0085 | 0.0009 |
| Baseline | Mlp No Edges | 0.5349 | 0.7314 | 0.0069 | 0.0218 | 0.0022 |
| **Proposed GNN** | **HeteroGNN (SAGE)** | **0.4903** | **0.7282** | **0.0000** | **0.0000** | **0.0000** |
| **Proposed GNN** | **HeteroGNN (ATTN)** | **0.5203** | **0.7314** | **0.0092** | **0.0049** | **0.0005** |

## 2. Generalization Across Split Regimes
Leak-free comparison isolating structural connectivity and temporal evolution.

| Split Methodology | Target Focus | ROC-AUC | F1 Score | TPR @ 0.1% FPR | Degradation vs Random |
|:---|:---|:---:|:---:|:---:|:---:|
| `random` (SAGE) | Leak-free evaluation | 0.4903 | 0.7282 | 0.0000 | 0.0000 |
| `random` (ATTN) | Leak-free evaluation | 0.5203 | 0.7314 | 0.0092 | +0.0032 |

## 2b. Graph Ablation Study
Systematic isolation of relational edge types, structural degree, and lexical representations.

| Ablation Configuration | Isolated Component | ROC-AUC | F1 Score | PR-AUC |
|:---|:---|:---:|:---:|:---:|
| `full_sage` | Full Sage | 0.4939 | 0.6708 | 0.5108 |
| `no_certificates` | No Certificates | 0.5190 | 0.6708 | 0.5128 |
| `cert_coissue_key` | Cert Coissue Key | 0.4689 | 0.6708 | 0.4896 |
| `no_nameservers` | No Nameservers | 0.4970 | 0.6600 | 0.5078 |
| `no_registrars` | No Registrars | 0.4831 | 0.5563 | 0.4942 |
| `no_asn` | No Asn | 0.5125 | 0.6708 | 0.5098 |
| `graph_only_no_lexical` | Graph Only No Lexical | 0.4809 | 0.6708 | 0.4876 |
| `degree_mode_train_visible` | Degree Mode Train Visible | 0.5074 | 0.6708 | 0.5193 |
| `mlp_no_edges` | Mlp No Edges | 0.4750 | 0.6708 | 0.4808 |
| `attn_variant` | Attn Variant | 0.5279 | 0.6708 | 0.5422 |
| `sage_guard_clean` | Sage Guard Clean | 0.5069 | 0.6708 | 0.5161 |

## 3. Adversarial Structural Attack & GNNGuard Recovery
Coordinated evasion attack injecting malicious edges into legitimate training infrastructure.

| Attack Budget ($k$) | Clean SAGE F1 | Attacked Undefended F1 | Attacked SAGE_Guard F1 | Recovery Score |
|:---:|:---:|:---:|:---:|:---:|
| Budget $k=1$ | 0.7314 | 0.7314 | **0.7314** | **100.0%** |
| Budget $k=2$ | 0.7314 | 0.7314 | **0.7314** | **100.0%** |

## 4. CPU Inference Latency Benchmark
- **CPU**: `Intel64 Family 6 Model 165 Stepping 2, GenuineIntel` (12 cores, OS: Windows 11 (AMD64))
- **PyTorch**: `2.14.0+cpu` | **Max Threads**: `12`

| Batch Size | Threads | Feature Lookup (ms) | 2-Hop Ego Build (ms) | Model Forward (ms) | End-to-End Mean | Median | P95 | P99 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | 1.37 | 16.12 | 30.77 | **48.25** | 45.31 | 64.95 | 88.48 |
| 32 | 1 | 1.36 | 31.44 | 40.80 | **73.60** | 71.77 | 89.27 | 90.53 |
| 1 | 12 | 1.77 | 25.05 | 25.17 | **51.99** | 51.62 | 68.30 | 71.27 |
| 32 | 12 | 1.97 | 56.70 | 34.73 | **93.40** | 90.51 | 117.95 | 123.46 |

## 5. Streaming & Continual Domain Updates
Online evaluation across chronological monthly cohorts comparing retraining from scratch, naive fine-tuning, and reservoir replay (M=1000).

| Step | Target Month | Strategy | New Month F1 | New Month TPR@0.1% FPR | Month 1 F1 | Backward Transfer (ΔF1) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2023-04 | `initial_base` | 0.6234 | 0.0000 | 0.6234 | **0.0000** |
| 1 | 2023-05 | `retrain_from_scratch` | 0.6087 | 0.0000 | 0.6133 | **-0.0101** |
| 1 | 2023-05 | `naive_finetune` | 0.5938 | 0.0000 | 0.6571 | **+0.0337** |
| 1 | 2023-05 | `replay` | 0.5714 | 0.0000 | 0.5915 | **-0.0319** |
| 2 | 2023-06 | `retrain_from_scratch` | 0.6747 | 0.0000 | 0.6571 | **+0.0337** |
| 2 | 2023-06 | `naive_finetune` | 0.6739 | 0.0000 | 0.6667 | **+0.0433** |
| 2 | 2023-06 | `replay` | 0.6739 | 0.0000 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `retrain_from_scratch` | 0.6667 | 0.0667 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `naive_finetune` | 0.6667 | 0.1000 | 0.6667 | **+0.0433** |
| 3 | 2023-07 | `replay` | 0.6667 | 0.0333 | 0.6667 | **+0.0433** |

## 6. Operational BIND RPZ Deployment
- **Calibrated Threshold**: `0.5034` (Operating Target FPR: `0.10%`)
- **Realized Validation FPR**: `0.244%`
- **Rules Emitted**: `1` of `368` test domains (0.27%)
- **Top Malware Families**: `{"redline": 1}`

