"""Baseline ML classifiers for jet tagging."""

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
import xgboost as xgb

from .classifier import Classifier


class MLClassifier(Classifier):
    """
    Base class for ML classifiers that operate on engineered features.

    This class handles feature extraction from raw particle data and
    delegates the actual classification to scikit-learn or XGBoost models.
    """

    def __init__(self, model):
        """
        Initialize ML classifier with a scikit-learn compatible model.

        Parameters
        ----------
        model : object
            A scikit-learn compatible model with fit/predict/predict_proba methods.
        """
        self.model = model
        self._is_fitted = False

    def _extract_features(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Extract features from raw particle data.

        Must be implemented by subclasses to define feature engineering strategy.

        Parameters
        ----------
        X : np.ndarray
            Raw jet data. Shape: (n_samples, n_particles, n_features).

        Returns
        -------
        np.ndarray
            Extracted features. Shape: (n_samples, n_extracted_features).
        """
        raise NotImplementedError("Subclasses must implement _extract_features")

    def fit(self, X: npt.NDArray[np.float32], y: npt.NDArray[np.int32]) -> "MLClassifier":
        """
        Train the classifier on extracted features.

        Parameters
        ----------
        X : np.ndarray
            Training data (jets). Shape: (n_samples, n_particles, n_features).
        y : np.ndarray
            Training labels. Shape: (n_samples,).

        Returns
        -------
        MLClassifier
            Returns self for method chaining.
        """
        X_features = self._extract_features(X)
        self.model.fit(X_features, y)
        self._is_fitted = True
        return self

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
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict")

        X_features = self._extract_features(X)
        return self.model.predict(X_features).astype(np.int32)

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Predict probability of quark jet (class 1).

        Parameters
        ----------
        X : np.ndarray
            Input data (jets). Shape: (n_samples, n_particles, n_features).

        Returns
        -------
        np.ndarray
            Predicted probabilities. Shape: (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba")

        X_features = self._extract_features(X)
        return self.model.predict_proba(X_features)[:, 1].astype(np.float32)


class MultiplicityLogisticRegression(MLClassifier):
    """
    Logistic regression on jet multiplicity (particle count).

    This is the simplest strong baseline. It uses only one feature:
    the number of particles in each jet (excluding padding).

    Typical performance: AUC ~0.67-0.70
    """

    def __init__(self, **kwargs):
        """
        Initialize multiplicity-based logistic regression.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to sklearn.linear_model.LogisticRegression.
            Defaults: random_state=42, max_iter=1000
        """
        model_kwargs = {"random_state": 42, "max_iter": 1000}
        model_kwargs.update(kwargs)
        super().__init__(SKLogisticRegression(**model_kwargs))

    def _extract_features(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Extract multiplicity feature (count of non-padded particles).

        Parameters
        ----------
        X : np.ndarray
            Raw jet data. Shape: (n_samples, n_particles, 4).

        Returns
        -------
        np.ndarray
            Multiplicity features. Shape: (n_samples, 1).
        """
        # Count particles with pt > 0 (non-padded)
        multiplicity = np.sum(X[:, :, 0] > 0, axis=1, keepdims=True)
        return multiplicity.astype(np.float32)


class XGBoostRawParticles(MLClassifier):
    """
    XGBoost classifier on flattened raw particle data.

    Uses the first 3 features (pt, rapidity, azimuthal_angle) of all particles,
    flattened into a fixed-size feature vector. Ignores pdgid to be realistic.

    Typical performance: AUC ~0.73-0.76
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 5, **kwargs):
        """
        Initialize XGBoost classifier on raw particle features.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds.
        max_depth : int
            Maximum tree depth.
        **kwargs : dict
            Additional arguments passed to xgboost.XGBClassifier.
        """
        model_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": 0.1,
            "random_state": 42,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        model_kwargs.update(kwargs)
        super().__init__(xgb.XGBClassifier(**model_kwargs))

    def _extract_features(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Flatten raw particle kinematics (pt, rapidity, phi) into feature vector.

        Parameters
        ----------
        X : np.ndarray
            Raw jet data. Shape: (n_samples, n_particles, 4).

        Returns
        -------
        np.ndarray
            Flattened features. Shape: (n_samples, n_particles * 3).
        """
        # Use only kinematic features (pt, rapidity, azimuthal_angle)
        # Exclude pdgid (index 3) to be realistic
        X_kinematic = X[:, :, :3]
        n_samples = X_kinematic.shape[0]
        return X_kinematic.reshape(n_samples, -1)
