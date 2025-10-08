"""Vibe Jet Tagging - LLM-based jet classification."""

from vibe_jet_tagging.classifier import Classifier
from vibe_jet_tagging.baselines import (
    MLClassifier,
    MultiplicityLogisticRegression,
    XGBoostRawParticles,
)

__all__ = [
    "Classifier",
    "MLClassifier",
    "MultiplicityLogisticRegression",
    "XGBoostRawParticles",
]


def hello() -> str:
    return "Hello from vibe-jet-tagging!"
