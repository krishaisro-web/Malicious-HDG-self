"""
Malicious-HDG: Heterogeneous Graph-Based Malicious Domain Classification.
Faithful schema modeling and leak-free evaluation for Zenodo DomainRadar v2.
"""

from src.hdg.models.hetero_gnn import HeteroGNN, GNNGuardLayer

__version__ = "2.0.0"
__all__ = ["HeteroGNN", "GNNGuardLayer"]
