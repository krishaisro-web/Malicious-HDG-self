"""
Training loops, optimization routines, and baseline models for Malicious-HDG.
"""

from src.hdg.training.train import train_eval_gnn
from src.hdg.training.baselines import (
    run_tfidf_lexical_baseline,
    run_xgboost_baseline,
    run_single_feature_baseline,
    run_mlp_no_edges_baseline,
)

__all__ = [
    "train_eval_gnn",
    "run_tfidf_lexical_baseline",
    "run_xgboost_baseline",
    "run_single_feature_baseline",
    "run_mlp_no_edges_baseline",
]
