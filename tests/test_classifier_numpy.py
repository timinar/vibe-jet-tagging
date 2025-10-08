"""Unit tests for Classifier base class with numpy arrays."""

import numpy as np
import pytest

from vibe_jet_tagging.classifier import Classifier


class DummyClassifier(Classifier):
    """Minimal concrete implementation for testing."""

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        if not hasattr(self, "fitted"):
            raise RuntimeError("Not fitted")
        return np.zeros(len(X), dtype=np.int32)

    def predict_proba(self, X):
        if not hasattr(self, "fitted"):
            raise RuntimeError("Not fitted")
        return np.full(len(X), 0.5, dtype=np.float32)


def test_classifier_is_abstract():
    """Test that Classifier cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Classifier()


def test_classifier_requires_fit_method():
    """Test that subclasses must implement fit()."""

    class IncompleteFit(Classifier):
        def predict(self, X):
            return np.zeros(len(X), dtype=np.int32)

    with pytest.raises(TypeError):
        IncompleteFit()


def test_classifier_requires_predict_method():
    """Test that subclasses must implement predict()."""

    class IncompletePredict(Classifier):
        def fit(self, X, y):
            return self

    with pytest.raises(TypeError):
        IncompletePredict()


def test_classifier_fit_returns_self():
    """Test that fit() returns self for method chaining."""
    clf = DummyClassifier()
    X = np.random.randn(10, 50, 4).astype(np.float32)
    y = np.random.randint(0, 2, 10, dtype=np.int32)

    result = clf.fit(X, y)
    assert result is clf


def test_classifier_predict_shape():
    """Test that predict() returns correct shape."""
    clf = DummyClassifier()
    X_train = np.random.randn(10, 50, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 10, dtype=np.int32)
    X_test = np.random.randn(5, 50, 4).astype(np.float32)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions.shape == (5,)
    assert predictions.dtype == np.int32


def test_classifier_predict_proba_shape():
    """Test that predict_proba() returns correct shape."""
    clf = DummyClassifier()
    X_train = np.random.randn(10, 50, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 10, dtype=np.int32)
    X_test = np.random.randn(5, 50, 4).astype(np.float32)

    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)

    assert probabilities.shape == (5,)
    assert probabilities.dtype == np.float32


def test_classifier_predict_proba_not_implemented_by_default():
    """Test that predict_proba() raises NotImplementedError by default."""

    class MinimalClassifier(Classifier):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.int32)

    clf = MinimalClassifier()
    X = np.random.randn(5, 50, 4).astype(np.float32)

    with pytest.raises(NotImplementedError, match="does not support predict_proba"):
        clf.predict_proba(X)


def test_classifier_predict_values():
    """Test that predict() returns binary labels."""
    clf = DummyClassifier()
    X_train = np.random.randn(10, 50, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 10, dtype=np.int32)
    X_test = np.random.randn(5, 50, 4).astype(np.float32)

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert np.all((predictions == 0) | (predictions == 1))


def test_classifier_predict_proba_range():
    """Test that predict_proba() returns values in [0, 1]."""
    clf = DummyClassifier()
    X_train = np.random.randn(10, 50, 4).astype(np.float32)
    y_train = np.random.randint(0, 2, 10, dtype=np.int32)
    X_test = np.random.randn(5, 50, 4).astype(np.float32)

    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
