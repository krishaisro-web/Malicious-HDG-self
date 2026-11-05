# Pre-Committed Empirical Analysis Plan: Malicious-HDG

**Date**: November 5, 2026  
**Status**: FROZEN PRIOR TO EVALUATION EXECUTION  
**Applicability**: Zenodo DomainRadar v2 (100k Domains Benchmark), ISRO Server Production Checkpoints  

---

## 1. Executive Protocol & Defense Against Forking Paths

To prevent p-hacking, post-hoc metric shopping, and the garden of forking paths, this document freezes all statistical comparisons, primary evaluation metrics, significance testing procedures, and decision rules before executing new evaluations or post-hoc additions.

The test set is **STRICTLY READ-ONLY**: exactly one evaluation pass per frozen configuration. All hyperparameter selections, early stopping, and threshold calibrations must occur exclusively on validation splits.

---

## 2. Target Metrics & Primary Objectives (O1–O5)

| Objective | Primary Metric | Target Criterion | Null Hypothesis / Acceptance Rule |
|:---|:---|:---|:---|
| **O1: Low-FPR Operation** | TPR @ 0.10% FPR ($\text{TPR}_{0.1\%}$) | $\text{TPR} \ge 0.50$ at realized test $\text{FPR} \le 0.10\%$ ($0.0010$) | Threshold calibrated on validation negatives ($\le 4$ FPs on $10,013$ pooled negatives; $\le 1$ FP on held-out cert split). Realized test FPR must satisfy exact 95% Clopper-Pearson CI upper bound $\le 0.0015$. |
| **O2: Temporal Generalization** | F1 Score Degradation ($\Delta \text{F1}_{R \to T}$) | $|\Delta \text{F1}| \le 0.10$ on chronological $T \to T+1$ | Evaluated with test month's structural edges visible (labels strictly hidden). DomainRadar contains only 1 valid multi-class month. |
| **O3: AUROC Superiority** | ROC-AUC | HeteroGNN AUROC $>$ Tabular XGBoost AUROC ($p < 0.05$) | Paired bootstrap difference on test predictions across 5 seeds; two-sided $p < 0.05$ required. |
| **O4: CPU Latency Budget** | Per-query end-to-end CPU inference time | Mean $< 50$ ms, P95 $< 50$ ms | Evaluated across batch sizes 1 and 32, on 1 thread and 32 threads. Pareto frontier with bounded neighbor sampling $[k_1, k_2]$. |
| **O5: Adversarial Robustness** | Three Robustness Differences & FPR ($\Delta_{\text{att}}, \Delta_{\text{res}}, \Delta_{\text{def}}$) | $\Delta_{\text{res}} \le 0.05$ with realized test $\text{FPR} \le 0.0015$ | Coordinated structural attack on purity-ranked target selection ($\text{degree} \times \text{train-benign purity} \ge 99\%$). Reported via paired differences with 95% bootstrap CIs: $\Delta_{\text{att}} = \text{F1}_{\text{clean}} - \text{F1}_{\text{att}}$, $\Delta_{\text{res}} = \text{F1}_{\text{clean}} - \text{F1}_{\text{def}}$, $\Delta_{\text{def}} = \text{F1}_{\text{def}} - \text{F1}_{\text{att}}$, alongside realized FPR under attack. (The ratio $(C-B)/(A-B)$ is rejected as degenerate when attack shifts F1 upward). |

---

## 3. Statistical Testing & Multiple-Comparison Corrections

### 3.1 Ablation Study Holm-Bonferroni Correction (Condition 1)
Across the 10 pairwise comparisons between the baseline (`full_sage`) and the 10 ablation variants on random split across 5 seeds:
- Nominal significance threshold: $\alpha = 0.05$.
- Procedure: Step-down Holm correction where the $i$-th smallest p-value is compared against $\frac{\alpha}{10 - i + 1}$.
- Pre-diagnosed outcomes:
  - **Surviving Correction ($p \le \alpha_{\text{Holm}}$)**:
    1. `mlp_no_edges` (extreme degradation without message passing)
    2. `no_registrars`
    3. `no_nameservers`
    4. `attn_variant` ($p = 0.0071$, threshold $\frac{0.05}{7} \approx 0.00714$, marginally significant)
  - **Non-Surviving ($p > \alpha_{\text{Holm}}$)**:
    5. `graph_only_no_lexical` (nominal $p = 0.0174 > 0.00833$; labeled *nominally significant, not robust to correction*)
    6. `degree_mode_train_visible` (nominal $p = 0.0312 > 0.0100$; labeled *nominally significant, not robust to correction*)
  - **Equivalence / Overhead Framing**:
    - For `sage_guard_clean`: A non-significant p-value is **not** evidence of zero overhead. Reported strictly as: *"no detectable overhead; 95% bootstrap CI bounds difference at $\le 0.011$ AUC (observed difference: $-0.0019$, 95% CI: $[-0.0108, +0.0070]$)"*.

### 3.2 ATTN Seed 777 Diagnosis (Condition 2)
- Seed 777 group-split log (`gnn_attn_group_seed_777.jsonl`) confirms validation ROC-AUC peaked at $0.7375$ at epoch 4, then did not improve for 20 epochs, triggering early stopping at epoch 24.
- Test ROC-AUC dropped to $0.5255$.
- **Diagnosis**: Model selection failed under group shift (validation–test gap 0.21; peak validation AUC was weak at 0.7375 compared to 0.8711–0.9069 on other seeds).
- Protocol rule: Seed 777 is **strictly retained** in all reported multi-seed means and standard deviations.
- On the remaining 4 seeds (42, 1337, 2024, 999), ATTN vs SAGE differences flip sign ($-0.0396, +0.0641, -0.0160, +0.0280$) with mean delta $+0.0091$. This is documented as *"no consistent difference"* across seeds.

### 3.3 Training Protocol Harmonization (Condition 3)
- Base SAGE in `04_train_gnn.py` achieved $0.9214 \pm 0.0057$, whereas `05_ablation.py` achieved $0.9293 \pm 0.0037$.
- Discrepancy etiology: `04` utilized a 5-epoch validation LR grid probe followed by long training (up to 200 epochs) with plateau learning rate decay, inducing overfit on some seeds, whereas `05` used a simpler fixed LR $0.005$, 30 epochs, and patience 10 ("reduced-budget protocol").
- Validation gating: All model selection, early stopping, and hyperparameters are determined strictly on validation ROC-AUC.
- Reporting rule: The ablation table is explicitly labeled as the *"reduced-budget protocol (30 epochs, fixed lr=0.005)"* to maintain absolute transparency. Protocol statements are aligned with actual code execution (early stopping on validation ROC-AUC).

### 3.4 Calibration & Certification Caveats (Condition 4)
- Certification at $\le 4$ FPs on $10,013$ validation negatives provides a **per-seed 95% confidence guarantee**. Across 5 independent evaluation seeds, the joint probability that all 5 test FPRs simultaneously fall $\le 0.10\%$ is $(0.95)^5 \approx 77.4\%$. Occasional single-seed excursions above $0.10\%$ are mathematically expected and reported with Clopper-Pearson CIs.
- Dual certification reporting:
  1. *Pooled validation calibration*: $\le 4$ FPs allowed on pooled validation negatives ($N=10,013$).
  2. *Held-out certification split*: 50/50 calibration/certification split ($\le 1$ FP allowed on held-out certification half).
- Operating comparison: SAGE 10.67% and ATTN 15.06% at matched FPR 0.094% are explicitly designated as **oracle-matched**, while deployable validation-calibrated TPRs are reported separately with bootstrap CIs.

---

## 4. Citation and Dataset Fact Verification (Condition 5)

1. **Theoretical Grounding for pAUC / Top-$k$ Loss**:
   - Grounded in partial AUC optimization theory: Boyd, Cortes, Mohri (2012) and Narasimhan & Agarwal (2013).
   - Related work: AO-GNN (Mengda Huang, Yang Liu, Xiang Ao, Kuan Li, Jianfeng Chi, Jinghua Feng, Yang Hao, and Qing He, "AUC-oriented Graph Neural Network for Fraud Detection", in *Proceedings of the ACM Web Conference 2022 (WWW '22)*, pp. 1311–1321, 2022) optimizes full ROC-AUC using AUC-oriented SGD and RL edge pruning under label imbalance, rather than partial AUC at low FPR; its edge-pruning mechanisms serve as background inspiration, not direct validation for top-$k$ surrogate losses.
2. **Dataset Count Attribution**:
   - CESNET benign record count: **461,388** in raw parquet/JSON extraction; **461,338** reported in the DomainRadar paper figure. This unexplained 50-record difference is explicitly noted with attribution rather than asserting unverified upstream deduplication.
3. **Ground-Truth Label Quality**:
   - Primary data source verification: ThreatFox, URLhaus, Firebog, and MISP threat feeds with VirusTotal verification as detailed in DomainRadar v2 paper (Hranický et al., "A Multi-Dimensional DNS Domain Intelligence Dataset for Cybersecurity Research", *Data in Brief*, Section 2.1 "Data Collection and Label Verification").

---

## 5. Pre-Committed Stop Rule (Condition 6)

For Step 3 (Post-Hoc Stacker with GBA features) and Step 4 (pAUC Fine-Tuning):
- **Decision Rule**: The additions are considered successful if and only if:
  $$\Delta \text{TPR}_{0.1\%} \ge +3.0 \text{ percentage points over baseline SAGE}$$
  with a 95% paired bootstrap confidence interval strictly excluding zero.
- **Negative Result Commit**: If the observed gain is $< +3.0$ pp or the 95% CI spans zero, it will be reported transparently as a **negative result**. The core thesis contribution is established by the FPR-anchored evaluation methodology, certified calibration gates, matched FPR comparisons, and leakage audits.

---
*Signed and frozen.*
