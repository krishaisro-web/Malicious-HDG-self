"""
Evaluation modules: Adversarial attack evaluation, CPU inference latency benchmark, and streaming continual learning.
"""

from src.hdg.eval.attack import generate_structural_attack, evaluate_attacked_model
from src.hdg.eval.latency import build_ego_subgraph, benchmark_cpu_query_latency
from src.hdg.eval.streaming import ReservoirBuffer, run_streaming_evaluation

__all__ = [
    "generate_structural_attack",
    "evaluate_attacked_model",
    "build_ego_subgraph",
    "benchmark_cpu_query_latency",
    "ReservoirBuffer",
    "run_streaming_evaluation",
]
