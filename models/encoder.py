"""
Backwards-compatibility shim for legacy encoder.
Canonical implementation resides in src.hdg.models.legacy.encoder.
"""
from src.hdg.models.legacy.encoder import (
    FeatureProjection,
    make_hetero_layer,
    ALL_RELATIONS,
    NODE_TYPES,
    IN_DIMS,
)

__all__ = ["FeatureProjection", "make_hetero_layer", "ALL_RELATIONS", "NODE_TYPES", "IN_DIMS"]