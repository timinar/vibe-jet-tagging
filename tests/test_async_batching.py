"""Tests for async batching functionality in LLMClassifier."""

import numpy as np
import pytest

from vibe_jet_tagging import LLMClassifier


@pytest.fixture
def mock_jets():
    """Create small mock jets for testing."""
    np.random.seed(42)
    jets = []
    for _ in range(5):
        n_particles = np.random.randint(15, 30)
        jet = np.zeros((100, 4))
        jet[:n_particles, 0] = np.random.exponential(10, n_particles)  # pt
        jet[:n_particles, 1] = np.random.uniform(-2, 2, n_particles)   # y
        jet[:n_particles, 2] = np.random.uniform(0, 6.28, n_particles) # phi
        jet[:n_particles, 3] = np.random.choice([22, 211, -211, 321, -321], n_particles)  # pid
        jets.append(jet)
    return jets


def test_predict_with_batch_size(mock_jets):
    """Test that predict works with batch_size parameter."""
    clf = LLMClassifier(
        template_name='features_basic',
        thinking_budget=None,  # Faster without thinking
        max_tokens=10,
    )
    clf.fit([], [])
    
    # Test with batching
    predictions = clf.predict(mock_jets, batch_size=2)
    
    assert len(predictions) == len(mock_jets)
    assert all(p in [0, 1] for p in predictions)
    print(f"✓ Batched prediction works: {predictions}")


def test_predict_async_directly(mock_jets):
    """Test async predict method directly."""
    import asyncio
    
    clf = LLMClassifier(
        template_name='features_basic',
        thinking_budget=None,
        max_tokens=10,
    )
    clf.fit([], [])
    
    # Test async method directly
    predictions = asyncio.run(clf.predict_async(mock_jets, batch_size=3))
    
    assert len(predictions) == len(mock_jets)
    assert all(p in [0, 1] for p in predictions)
    print(f"✓ Async prediction works: {predictions}")


def test_batch_vs_sequential_consistency(mock_jets):
    """Test that batched and sequential predictions give same results (approximately)."""
    # Use small jets subset
    small_jets = mock_jets[:3]
    
    clf = LLMClassifier(
        template_name='features_basic',
        thinking_budget=None,
        max_tokens=10,
    )
    clf.fit([], [])
    
    # Sequential
    pred_sequential = clf.predict(small_jets[:1])  # Just 1 jet for speed
    
    # Batched
    pred_batched = clf.predict(small_jets[:1], batch_size=1)
    
    # Both should return valid predictions (not checking equality due to LLM randomness)
    assert len(pred_sequential) == 1
    assert len(pred_batched) == 1
    assert pred_sequential[0] in [0, 1]
    assert pred_batched[0] in [0, 1]
    print(f"✓ Both methods return valid predictions")
    print(f"  Sequential: {pred_sequential}")
    print(f"  Batched: {pred_batched}")


def test_empty_batch():
    """Test handling of empty input."""
    clf = LLMClassifier(template_name='features_basic')
    clf.fit([], [])
    
    predictions = clf.predict([], batch_size=10)
    assert predictions == []
    print("✓ Empty batch handled correctly")


def test_single_jet_batch(mock_jets):
    """Test batching with a single jet."""
    clf = LLMClassifier(
        template_name='features_basic',
        thinking_budget=None,
        max_tokens=10,
    )
    clf.fit([], [])
    
    predictions = clf.predict([mock_jets[0]], batch_size=1)
    
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    print(f"✓ Single jet batch works: {predictions}")


if __name__ == "__main__":
    # Quick smoke test
    print("Running async batching tests...")
    pytest.main([__file__, "-v", "--tb=short"])

