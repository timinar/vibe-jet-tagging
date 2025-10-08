"""Vibe Jet Tagging - LLM-based jet classification."""

from vibe_jet_tagging.classifier import Classifier
from vibe_jet_tagging.llm_classifier import LLMClassifier

__all__ = ["Classifier", "LLMClassifier"]


def hello() -> str:
    return "Hello from vibe-jet-tagging!"
