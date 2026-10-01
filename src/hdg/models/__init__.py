"""
Models package for Malicious-HDG.
Exposes HeteroGNN and GNNGuardLayer architectures.
"""

from src.hdg.models.hetero_gnn import HeteroGNN, GNNGuardLayer

__all__ = ["HeteroGNN", "GNNGuardLayer"]
