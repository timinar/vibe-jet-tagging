"""Base classifier interface for jet tagging."""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class Classifier(ABC):
    """
    Abstract base class for jet classifiers.

    All classifiers must implement fit() and predict() methods.
    Input/output uses numpy arrays matching the dataset format.
    """

    @abstractmethod
    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int32]) -> "Classifier":
        """
        Train or prepare the classifier.

        Parameters
        ----------
        X : np.ndarray
            Training data (jets). Shape: (n_samples, n_particles, n_features).
            Features are [pt, rapidity, azimuthal_angle, pdgid] per particle.
        y : np.ndarray
            Training labels. Shape: (n_samples,).
            1 for quark jets, 0 for gluon jets.

        Returns
        -------
        Classifier
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
        """
        Predict jet labels.

        Parameters
        ----------
        X : np.ndarray
            Input data (jets). Shape: (n_samples, n_particles, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels. Shape: (n_samples,).
            1 for quark jets, 0 for gluon jets.
        """
        pass

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Predict class probabilities (optional, for classifiers that support it).

        Parameters
        ----------
        X : np.ndarray
            Input data (jets). Shape: (n_samples, n_particles, n_features).

        Returns
        -------
        np.ndarray
            Predicted probabilities. Shape: (n_samples,).
            Probability of being a quark jet (class 1).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba"
        )

