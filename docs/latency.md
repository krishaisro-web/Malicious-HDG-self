# CPU Inference Latency Benchmark Protocol

## 1. Overview & Operational Motivation

Real-time deployment of domain maliciousness classifiers (e.g., at DNS recursive resolvers, perimeter security gateways, or threat-intelligence ingestion pipelines) demands sub-second per-query classification latency. While graph neural networks leverage rich structural connectivity, real-time query latency must remain operationally practical on standard enterprise CPU servers without requiring dedicated GPU accelerators.

This benchmark measures the exact latency breakdown for single-query and batch classification on CPU.

---

## 2. Benchmark Measurement Stages

The evaluation isolates three distinct stages of the inference pipeline:

1. **Feature Extraction & Standardization (`feature_lookup`)**:
   - Extracting 12 technical domain attributes (DNS response flags, TTL values, MX counts, DNSSEC, RDAP presence, TLS certificate parameters).
   - Computing 5 lexical statistics (domain length, character entropy, digit ratio, vowel ratio, longest consonant sequence).
   - Transforming the resulting vector using the pre-fit `TrainOnlyStandardizer` ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$).

2. **Model Forward Pass (`forward_pass`)**:
   - Executing the PyG `HeteroGNN` forward computation (linear feature projection, multi-relation message passing layers, and classification head MLP).
   - Computing normalized softmax probabilities on CPU.

3. **End-to-End Query Latency (`end_to_end`)**:
   - Complete elapsed wall-clock time from raw domain ingestion to final calibrated malicious score.

---

## 3. Execution Protocol & Parameters

To ensure statistical reliability and eliminate cold-start artifacts:
- **High-Precision Timing**: Measurements use Python's monotonic high-resolution timer: `time.perf_counter()`.
- **Warmup Iterations**: 10 full forward passes are executed prior to measurement to warm CPU instruction caches, memory pools, and thread buffers.
- **Sample Size**: 100 queries sampled across test domains.
- **Reported Statistics**:
  - **Mean ($ms$)**: Arithmetic average latency across queries.
  - **Median ($ms$)**: 50th percentile (typical operating latency).
  - **P95 ($ms$)**: 95th percentile latency (tail latency under system load).
  - **Min / Max ($ms$)**: Observed bounds.

---

## 4. Execution Command

To run the benchmark:
```bash
# On synthetic fixture:
python -m src.scripts.07_latency_eval --fixture --num-queries 100 --warmup 10

# On real dataset:
python -m src.scripts.07_latency_eval --num-queries 100 --warmup 10
```
Artifacts are automatically recorded in:
- `results/tables/latency_results.json`
- `results/tables/latency_results.md`
