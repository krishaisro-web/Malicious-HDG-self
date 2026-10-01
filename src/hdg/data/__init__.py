"""
Data processing, schema parsing, graph construction, and split management for Malicious-HDG.
"""

from src.hdg.data.parse import parse_and_process_dataset, stream_parse_file
from src.hdg.data.profiler import profile_dataset
from src.hdg.data.audit import run_shortcut_audit
from src.hdg.data.splits import (
    DataSplits,
    random_stratified_split,
    rolling_origin_temporal_splits,
    time_split,
    group_split_bipartite,
    group_split_by_asn,
    compute_asn_leakage_report,
)
from src.hdg.data.graph import (
    build_base_graph,
    apply_split_scaling,
    build_hetero_graph,
    TrainOnlyStandardizer,
    extract_domain_feature_vector,
    extract_domain_features_vectorized,
)
from src.hdg.data.fixture import make_fixture

__all__ = [
    "parse_and_process_dataset",
    "stream_parse_file",
    "profile_dataset",
    "run_shortcut_audit",
    "DataSplits",
    "random_stratified_split",
    "rolling_origin_temporal_splits",
    "time_split",
    "group_split_bipartite",
    "group_split_by_asn",
    "compute_asn_leakage_report",
    "build_base_graph",
    "apply_split_scaling",
    "extract_domain_feature_vector",
    "extract_domain_features_vectorized",
    "make_fixture",
]
