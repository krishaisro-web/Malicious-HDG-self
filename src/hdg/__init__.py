"""
Malicious-HDG: Heterogeneous Graph-Based Malicious Domain Classification.
Faithful schema modeling and leak-free evaluation for Zenodo DomainRadar v2.
"""

from src.hdg.config import load_config, get_resolved_paths, init_thread_pool, ResolvedPaths
from src.hdg.metrics import (
    evaluate_predictions,
    choose_optimal_threshold,
    compute_bootstrap_cis,
    compute_prevalence_adjusted_precision,
    paired_mcnemar_test,
    paired_bootstrap_auc_difference,
    multi_seed_paired_comparisons,
)
from src.hdg.models.hetero_gnn import HeteroGNN, GNNGuardLayer

# Data subpackage re-exports for backwards compatibility
from src.hdg.data.parse import parse_and_process_dataset, stream_parse_file
from src.hdg.data.profiler import profile_dataset
from src.hdg.data.audit import run_shortcut_audit
from src.hdg.data.splits import DataSplits
from src.hdg.data.graph import build_base_graph, apply_split_scaling, build_hetero_graph, TrainOnlyStandardizer
from src.hdg.data.fixture import make_fixture

# Training subpackage re-exports
from src.hdg.training.train import train_eval_gnn
from src.hdg.training.baselines import (
    run_tfidf_lexical_baseline,
    run_xgboost_baseline,
    run_single_feature_baseline,
    run_mlp_no_edges_baseline,
)

# Eval subpackage re-exports
from src.hdg.eval.attack import generate_structural_attack, evaluate_attacked_model
from src.hdg.eval.latency import build_ego_subgraph, benchmark_cpu_query_latency
from src.hdg.eval.streaming import ReservoirBuffer, run_streaming_evaluation

__version__ = "2.0.0"

__all__ = [
    "load_config",
    "get_resolved_paths",
    "init_thread_pool",
    "ResolvedPaths",
    "evaluate_predictions",
    "choose_optimal_threshold",
    "compute_bootstrap_cis",
    "compute_prevalence_adjusted_precision",
    "paired_mcnemar_test",
    "paired_bootstrap_auc_difference",
    "multi_seed_paired_comparisons",
    "HeteroGNN",
    "GNNGuardLayer",
    "parse_and_process_dataset",
    "stream_parse_file",
    "profile_dataset",
    "run_shortcut_audit",
    "DataSplits",
    "build_base_graph",
    "apply_split_scaling",
    "make_fixture",
    "train_eval_gnn",
    "run_tfidf_lexical_baseline",
    "run_xgboost_baseline",
    "run_single_feature_baseline",
    "run_mlp_no_edges_baseline",
    "generate_structural_attack",
    "evaluate_attacked_model",
    "build_ego_subgraph",
    "benchmark_cpu_query_latency",
    "ReservoirBuffer",
    "run_streaming_evaluation",
]
