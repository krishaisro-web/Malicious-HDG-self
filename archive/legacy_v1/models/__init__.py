"""
Backwards-compatibility shim for Malicious-HDG models.
Canonical models reside in src.hdg.models.
"""
from src.hdg.models import HeteroGNN, GNNGuardLayer

__all__ = ["HeteroGNN", "GNNGuardLayer"]
