"""
Legacy model components for Malicious-HDG (FullModel, FeatureProjection, Classifier).
Retained for backwards-compatibility with exploratory snapshot scripts.
"""

from src.hdg.models.legacy.full_model import FullModel
from src.hdg.models.legacy.classifier import Classifier
from src.hdg.models.legacy.encoder import FeatureProjection, make_hetero_layer, ALL_RELATIONS
from src.hdg.models.legacy.temporal import TemporalCombiner

__all__ = [
    "FullModel",
    "Classifier",
    "FeatureProjection",
    "make_hetero_layer",
    "ALL_RELATIONS",
    "TemporalCombiner",
]
