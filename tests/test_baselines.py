"""Integration tests for baseline classifiers."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, accuracy_score

from vibe_jet_tagging.baselines import (
    MultiplicityLogisticRegression,
    XGBoostRawParticles,
)


@pytest.fixture
def synthetic_jet_data():
    """Create synthetic jet data mimicking the real dataset structure."""
    np.random.seed(42)
    n_samples = 200
    n_particles = 50
    n_features = 4

    # Create quark jets (fewer particles, higher pt concentration)
    n_quark = n_samples // 2
    quark_multiplicity = 30
    X_quark = np.zeros((n_quark, n_particles, n_features), dtype=np.float32)
    for i in range(n_quark):
        n_real = quark_multiplicity + np.random.randint(-5, 5)
        X_quark[i, :n_real, 0] = np.random.exponential(2.0, n_real)  # pt
        X_quark[i, :n_real, 1] = np.random.randn(n_real) * 0.5  # rapidity
        X_quark[i, :n_real, 2] = np.random.uniform(0, 2 * np.pi, n_real)  # phi
        X_quark[i, :n_real, 3] = np.random.choice([211, -211, 111], n_real)  # pdgid

    # Create gluon jets (more particles, more diffuse)
    n_gluon = n_samples // 2
    gluon_multiplicity = 45
    X_gluon = np.zeros((n_gluon, n_particles, n_features), dtype=np.float32)
    for i in range(n_gluon):
        n_real = gluon_multiplicity + np.random.randint(-5, 5)
        X_gluon[i, :n_real, 0] = np.random.exponential(1.5, n_real)  # pt (softer)
        X_gluon[i, :n_real, 1] = np.random.randn(n_real) * 0.7  # rapidity (wider)
        X_gluon[i, :n_real, 2] = np.random.uniform(0, 2 * np.pi, n_real)  # phi
        X_gluon[i, :n_real, 3] = np.random.choice([211, -211, 111], n_real)  # pdgid

    # Combine and shuffle
    X = np.vstack([X_quark, X_gluon])
    y = np.hstack([np.ones(n_quark, dtype=np.int32), np.zeros(n_gluon, dtype=np.int32)])

    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def test_multiplicity_lr_fit_predict(synthetic_jet_data):
    """Test MultiplicityLogisticRegression can fit and predict."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = MultiplicityLogisticRegression()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions.shape == (50,)
    assert predictions.dtype == np.int32
    assert np.all((predictions == 0) | (predictions == 1))


def test_multiplicity_lr_predict_proba(synthetic_jet_data):
    """Test MultiplicityLogisticRegression predict_proba."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = MultiplicityLogisticRegression()
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)

    assert probabilities.shape == (50,)
    assert probabilities.dtype == np.float32
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_multiplicity_lr_performance(synthetic_jet_data):
    """Test that MultiplicityLogisticRegression achieves reasonable AUC."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = MultiplicityLogisticRegression()
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)

    auc = roc_auc_score(y_test, y_prob)
    # Should be significantly better than random (0.5)
    # Synthetic data has strong multiplicity signal
    assert auc > 0.7, f"AUC {auc:.3f} is too low for multiplicity baseline"


def test_multiplicity_lr_method_chaining(synthetic_jet_data):
    """Test that fit returns self for method chaining."""
    X, y = synthetic_jet_data

    clf = MultiplicityLogisticRegression()
    result = clf.fit(X, y)

    assert result is clf


def test_xgboost_fit_predict(synthetic_jet_data):
    """Test XGBoostRawParticles can fit and predict."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = XGBoostRawParticles(n_estimators=10, max_depth=3)  # Small for speed
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions.shape == (50,)
    assert predictions.dtype == np.int32
    assert np.all((predictions == 0) | (predictions == 1))


def test_xgboost_predict_proba(synthetic_jet_data):
    """Test XGBoostRawParticles predict_proba."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = XGBoostRawParticles(n_estimators=10, max_depth=3)
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)

    assert probabilities.shape == (50,)
    assert probabilities.dtype == np.float32
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_xgboost_performance(synthetic_jet_data):
    """Test that XGBoostRawParticles achieves reasonable AUC."""
    X, y = synthetic_jet_data
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    clf = XGBoostRawParticles(n_estimators=50, max_depth=4)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)

    auc = roc_auc_score(y_test, y_prob)
    # XGBoost should perform well on synthetic data
    assert auc > 0.75, f"AUC {auc:.3f} is too low for XGBoost baseline"


def test_xgboost_uses_kinematic_features_only(synthetic_jet_data):
    """Test that XGBoost excludes pdgid (feature index 3)."""
    X, y = synthetic_jet_data
    n_particles = X.shape[1]

    clf = XGBoostRawParticles(n_estimators=10)
    features = clf._extract_features(X[:10])

    # Should have 3 features per particle (pt, rapidity, phi), not 4
    expected_shape = (10, n_particles * 3)
    assert features.shape == expected_shape


def test_multiplicity_extracts_correct_feature(synthetic_jet_data):
    """Test that multiplicity feature extraction is correct."""
    X, y = synthetic_jet_data

    clf = MultiplicityLogisticRegression()
    features = clf._extract_features(X[:10])

    # Should be (10, 1) - one feature per sample
    assert features.shape == (10, 1)

    # Manually verify multiplicity for first sample
    expected_mult = np.sum(X[0, :, 0] > 0)
    assert features[0, 0] == expected_mult


def test_predict_before_fit_raises_error(synthetic_jet_data):
    """Test that predict/predict_proba raise error before fit."""
    X, y = synthetic_jet_data

    clf_lr = MultiplicityLogisticRegression()
    clf_xgb = XGBoostRawParticles()

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf_lr.predict(X)

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf_lr.predict_proba(X)

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf_xgb.predict(X)

    with pytest.raises(RuntimeError, match="must be fitted"):
        clf_xgb.predict_proba(X)


def test_classifiers_handle_variable_multiplicity(synthetic_jet_data):
    """Test that classifiers handle jets with different particle counts."""
    X, y = synthetic_jet_data

    # Verify we have variable multiplicities
    multiplicities = np.sum(X[:, :, 0] > 0, axis=1)
    assert multiplicities.min() < multiplicities.max()

    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    # Both classifiers should handle this fine
    clf_lr = MultiplicityLogisticRegression()
    clf_lr.fit(X_train, y_train)
    pred_lr = clf_lr.predict(X_test)

    clf_xgb = XGBoostRawParticles(n_estimators=10)
    clf_xgb.fit(X_train, y_train)
    pred_xgb = clf_xgb.predict(X_test)

    assert len(pred_lr) == len(X_test)
    assert len(pred_xgb) == len(X_test)
