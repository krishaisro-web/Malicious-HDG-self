# Adversarial Structural Attack & GNNGuard Recovery
*Multi-instance PPT threat model: Attacker injects fake edges to top benign infrastructure (ranked by degree * train-benign purity >= 99%)*
*Evaluated across 5 seeded runs (mean ± std) with paired bootstrap 95% CIs*

| Budget (k) | Clean SAGE F1 (FPR) | Attacked Undefended F1 (FPR) | Attacked + GNNGuard F1 (FPR) | Δ_att [95% CI] | Residual Gap [95% CI] | Defended Gain [95% CI] | Recovery Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| k = 1 | 0.8025 (0.000%) | 0.8033 (0.000%) | 0.7668 (0.000%) | +0.0008 [-0.0027, +0.0043] | -0.0357 [-0.0657, -0.0035] | -0.0365 [-0.0677, -0.0030] | 417.1% |
| k = 2 | 0.8025 (0.000%) | 0.8499 (0.000%) | 0.8104 (0.000%) | +0.0474 [+0.0440, +0.0503] | +0.0079 [-0.0206, +0.0306] | -0.0396 [-0.0683, -0.0169] | 82.1% |
| k = 5 | 0.8025 (0.000%) | 0.8743 (0.000%) | 0.8587 (0.000%) | +0.0718 [+0.0668, +0.0756] | +0.0562 [+0.0414, +0.0710] | -0.0156 [-0.0313, -0.0025] | 22.1% |
| k = 10 | 0.8025 (0.000%) | 0.8741 (0.000%) | 0.8730 (0.000%) | +0.0716 [+0.0655, +0.0757] | +0.0705 [+0.0631, +0.0772] | -0.0011 [-0.0075, +0.0053] | 1.4% |

> **Metric O5 Definitions (Unified Reviewer Sign Convention: m_B - m_A, m_C - m_A, m_C - m_B)**:
> 1. **Δ_att (Attack Impact)**: m_B - m_A (Attacked Undefended F1 - Clean F1; negative indicates evasion degradation)
> 2. **Residual Gap**: m_C - m_A (Attacked Defended F1 - Clean F1; negative indicates gap relative to clean baseline)
> 3. **Defended Gain**: m_C - m_B (Attacked Defended F1 - Attacked Undefended F1; positive indicates defense recovery)
> Algebraic identity: Residual Gap = Attack Impact + Defended Gain.
