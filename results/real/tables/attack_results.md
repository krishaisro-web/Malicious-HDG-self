# Adversarial Structural Attack & GNNGuard Recovery
*Multi-instance PPT threat model: Attacker injects fake edges to top benign infrastructure*
*Evaluated across 5 seeded runs (mean ± std)*

| Budget (k) | (A) Clean SAGE F1 | (B) Attacked Undefended F1 | (C) Attacked + GNNGuard F1 | Evasion Rate | GNNGuard Recovery |
|---|---|---|---|---|---|
| k = 1 | 0.8025 ± 0.0061 | 0.8033 ± 0.0101 | 0.7668 ± 0.0326 | 6.9% | 417.1% |
| k = 2 | 0.8025 ± 0.0061 | 0.8499 ± 0.0080 | 0.8104 ± 0.0265 | 1.6% | 82.1% |
| k = 5 | 0.8025 ± 0.0061 | 0.8743 ± 0.0049 | 0.8587 ± 0.0118 | 0.1% | 22.1% |
| k = 10 | 0.8025 ± 0.0061 | 0.8741 ± 0.0045 | 0.8730 ± 0.0028 | 0.0% | 1.4% |

> **Metric O5 (Robustness)**: Recovery is computed as (C - B) / (A - B). Values >= 100% indicate full mitigation of injected adversarial edges.
