# CPU Inference Latency Benchmark Protocol

## 1. Overview & Operational Motivation

Real-time deployment of domain maliciousness classifiers (e.g., at DNS recursive resolvers, perimeter security gateways, or threat-intelligence ingestion pipelines) demands sub-second per-query classification latency. While graph neural networks leverage rich structural connectivity, real-time query latency must remain operationally practical on standard enterprise CPU servers without requiring dedicated GPU accelerators.

In real-world DNS resolution environments, an incoming query for an unclassified domain cannot wait for an entire 250,000-node graph forward pass. Instead, an operational system extracts the local structural neighborhood around the query domain on-the-fly and executes inference strictly on this localized sub-network.

This benchmark measures the exact latency breakdown for single-query ($B=1$) and mini-batch ($B=32$) classification on CPU across both single-thread and multi-thread configurations.

---

## 2. Receptive Field & Mathematical Equivalence of 2-Hop Ego-Subgraphs

For an $L$-layer Message Passing Neural Network (MPNN) such as HeteroGNN (`HeteroSAGE` or `HeteroGAT`), the representation of any target node $u \in \mathcal{V}$ after $L$ message passing steps is strictly a function of nodes within its $L$-hop computational graph:
$$\mathbf{h}_u^{(L)} = \text{AGGREGATE}\left(\left\{\mathbf{h}_v^{(L-1)} : v \in \mathcal{N}(u) \cup \{u\}\right\}\right)$$

Because Malicious-HDG employs an $L=2$ layer architecture:
1. **Target Node**: The query domain $u$.
2. **1-Hop Neighborhood**: Infrastructure directly connected to $u$ (`resolves_to` IP, `uses_ns` Nameserver, `registered_by` Registrar, `uses_cert` Certificate).
3. **2-Hop Neighborhood**: Infrastructure reachable via 1 intermediate hop (e.g., `ip -> belongs_to_asn -> asn`, `nameserver -> ns_ip -> ip`, and reverse edges back to co-hosted sibling domains).

### Mathematical Equivalence Guarantee:
Let $G = (\mathcal{V}, \mathcal{E}, \mathbf{X})$ be the full heterogeneous graph, and let $G_u = (\mathcal{V}_u, \mathcal{E}_u, \mathbf{X}_u)$ be the 2-hop induced ego-subgraph centered at $u$, with node features $\mathbf{X}_u = \mathbf{X}[\mathcal{V}_u]$. Then for any parameter set $\Theta$:
$$\text{HeteroGNN}(G_u; \Theta)[u] \equiv \text{HeteroGNN}(G; \Theta)[u]$$
up to standard floating-point roundoff error ($\Delta < 10^{-6}$).

Benchmarking inference latency on the true 2-hop ego-subgraph faithfully reflects production deployment latency rather than the unrealistic overhead of slicing a 250,000-node full-graph forward pass.

---

## 3. Benchmark Measurement Stages

The evaluation isolates four distinct stages of the inference pipeline:

1. **Feature Lookup & Standardization (`feature_lookup`)**:
   - Extracting 12 technical domain attributes (DNS response flags, TTL values, MX counts, DNSSEC, RDAP presence, TLS certificate parameters).
   - Computing 5 lexical statistics (domain length, character entropy, digit ratio, vowel ratio, longest consonant sequence).
   - Transforming the resulting vector using the pre-fit `TrainOnlyStandardizer` ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$).

2. **2-Hop Ego-Subgraph Extraction (`ego_subgraph_build`)**:
   - Querying the local topological adjacency for domain $u$.
   - Extracting incident IPs, NS, Registrars, Certs, and second-hop ASNs.
   - Constructing a compact `torch_geometric.data.HeteroData` mini-batch containing only reachable nodes and localized edge indices.

3. **Model Forward Pass (`model_forward`)**:
   - Executing the PyG `HeteroGNN` forward computation on CPU across the local ego-subgraph.
   - Computing normalized softmax probabilities for the query domain(s).

4. **End-to-End Query Latency (`end_to_end`)**:
   - Complete elapsed wall-clock time from raw query domain identifier to final malicious classification score.

---

## 4. Execution Protocol & Parameters

To ensure statistical reliability and eliminate cold-start artifacts:
- **High-Precision Timing**: Measurements use Python's monotonic high-resolution timer: `time.perf_counter()`.
- **Warmup Iterations**: $\ge 20$ full forward passes are executed prior to measurement to warm CPU instruction caches, memory pools, and thread buffers.
- **Sample Size**: $\ge 500$ queries sampled across test domains (50 in smoke/fixture mode).
- **Concurrency & Threading**:
  - Evaluated at `threads=1` (single-thread deployment model for high-throughput microservices).
  - Evaluated at `threads=N` (all available physical/logical CPU cores via OpenMP).
- **Batching**:
  - `Batch Size = 1`: Point-lookup latency for real-time DNS resolver inline inspection.
  - `Batch Size = 32`: Mini-batch throughput for near-line security gateway ingestion.
- **Reported Statistics**:
  - **Mean ($ms$)**: Arithmetic average latency across queries.
  - **Median ($ms$)**: 50th percentile (typical operating latency).
  - **P95 ($ms$)**: 95th percentile latency (tail latency under system load).
  - **P99 ($ms$)**: 99th percentile tail latency.

---

## 5. Execution Command

To run the benchmark:
```bash
# On synthetic fixture:
python -m src.scripts.07_latency_eval --fixture --smoke --force

# On real production dataset:
python -m src.scripts.07_latency_eval --num-queries 500 --warmup 20 --force
```

Artifacts are automatically recorded in:
- `results/tables/latency_results.json`
- `results/tables/latency_results.md`
- `results/runs/latency_eval.json`
