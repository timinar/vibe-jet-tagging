"""Tests for the base Classifier class."""

import pytest

from vibe_jet_tagging import Classifier


class DummyClassifier(Classifier):
    """Concrete implementation for testing."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X, y):
        """Store training data size."""
        self.is_fitted = True
        self.n_samples_ = len(X)
        return self

    def predict(self, X):
        """Return all ones."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        return [1] * len(X)


def test_classifier_is_abstract():
    """Test that Classifier cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Classifier()


def test_dummy_classifier_fit():
    """Test that fit works and returns self."""
    clf = DummyClassifier()
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = [0, 1, 0]
    
    result = clf.fit(X, y)
    
    assert result is clf  # Returns self
    assert clf.is_fitted
    assert clf.n_samples_ == 3


def test_dummy_classifier_predict():
    """Test that predict returns correct type and values."""
    clf = DummyClassifier()
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = [0, 1, 1]
    clf.fit(X_train, y_train)
    
    X_test = [[7, 8], [9, 10]]
    predictions = clf.predict(X_test)
    
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert all(isinstance(p, int) for p in predictions)
    assert all(p in [0, 1] for p in predictions)


def test_dummy_classifier_predict_before_fit():
    """Test that predict raises error if called before fit."""
    clf = DummyClassifier()
    X = [[1, 2], [3, 4]]
    
    with pytest.raises(ValueError, match="must be fitted"):
        clf.predict(X)


def test_classifier_with_jet_data():
    """Test classifier with realistic jet data as list of jets."""
    clf = DummyClassifier()
    
    # Simulate jet data: list of jets (each jet can be any format)
    X = [
        {"particles": [[0.5, 0.2, 1.2, 211], [0.3, -0.1, 2.1, -211]]},
        {"particles": [[1.2, 0.5, 0.8, 2212], [0.1, 0.0, 1.5, 211]]},
        {"particles": [[0.8, -0.3, 2.5, 111]]},
    ]
    y = [1, 0, 1]
    
    clf.fit(X, y)
    predictions = clf.predict(X)
    
    assert len(predictions) == 3
    assert all(p in [0, 1] for p in predictions)

