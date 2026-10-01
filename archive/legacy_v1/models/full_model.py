"""
Backwards-compatibility shim for legacy FullModel.
Canonical implementation resides in src.hdg.models.legacy.full_model.
"""
from src.hdg.models.legacy.full_model import FullModel
from src.hdg.models.legacy.encoder import FeatureProjection, make_hetero_layer, ALL_RELATIONS
from src.hdg.models.legacy.classifier import Classifier

__all__ = ["FullModel", "FeatureProjection", "make_hetero_layer", "ALL_RELATIONS", "Classifier"]