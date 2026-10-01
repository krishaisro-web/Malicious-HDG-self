# [FIXTURE - meaningless] Adversarial Structural Attack & GNNGuard Recovery
*Multi-instance PPT threat model: Attacker injects fake edges to top benign infrastructure*
*Evaluated across 1 seeded runs (mean ± std)*

| Budget (k) | (A) Clean SAGE F1 | (B) Attacked Undefended F1 | (C) Attacked + GNNGuard F1 | Evasion Rate | GNNGuard Recovery |
|---|---|---|---|---|---|
| k = 1 | 0.7314 ± 0.0000 | 0.7314 ± 0.0000 | 0.7314 ± 0.0000 | 0.0% | 100.0% |
| k = 2 | 0.7314 ± 0.0000 | 0.7314 ± 0.0000 | 0.7314 ± 0.0000 | 0.0% | 100.0% |

> **Metric O5 (Robustness)**: Recovery is computed as (C - B) / (A - B). Values >= 100% indicate full mitigation of injected adversarial edges.
