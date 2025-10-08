"""Base classifier interface for jet tagging."""

from abc import ABC, abstractmethod
from typing import Any


class Classifier(ABC):
    """
    Abstract base class for jet classifiers.
    
    All classifiers must implement fit() and predict() methods.
    """

    @abstractmethod
    def fit(self, X: list[Any], y: list[Any]) -> "Classifier":
        """
        Train or prepare the classifier.

        Parameters
        ----------
        X : list
            Training data (jets). Format depends on classifier implementation.
        y : list
            Training labels. 1 for quark jets, 0 for gluon jets.

        Returns
        -------
        Classifier
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: list[Any]) -> list[int]:
        """
        Predict jet labels.

        Parameters
        ----------
        X : list
            Input data (jets). Format depends on classifier implementation.

        Returns
        -------
        list[int]
            Predicted labels. 1 for quark jets, 0 for gluon jets.
        """
        pass

