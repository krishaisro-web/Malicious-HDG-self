"""
Backwards-compatibility shim for HeteroGNN and GNNGuardLayer.
Canonical models reside in src.hdg.models.hetero_gnn.
"""
from src.hdg.models.hetero_gnn import HeteroGNN, GNNGuardLayer

__all__ = ["HeteroGNN", "GNNGuardLayer"]
