"""
Test LLMClassifier with Google Gemini API.

Tests thinking_budget control and basic classification functionality.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vibe_jet_tagging import LLMClassifier


@pytest.fixture
def sample_jet():
    """Load a single sample jet for testing."""
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'
    data = np.load(data_path)
    return data['X'][0]


@pytest.fixture
def sample_jets():
    """Load 10 sample jets for testing."""
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'
    data = np.load(data_path)
    return data['X'][:10], data['y'][:10]


def test_llm_classifier_init():
    """Test LLMClassifier initialization."""
    clf = LLMClassifier(
        model_name="gemini-2.5-flash-lite-preview-09-2025",
        template_name="simple_list",
        format_type="list",
        thinking_budget=1000
    )
    
    assert clf.model_name == "gemini-2.5-flash-lite-preview-09-2025"
    assert clf.thinking_budget == 1000
    assert not clf.is_fitted


def test_llm_classifier_fit():
    """Test LLMClassifier fit method."""
    clf = LLMClassifier()
    clf.fit([], [])
    
    assert clf.is_fitted


def test_llm_classifier_predict_single(sample_jet):
    """Test prediction on a single jet."""
    clf = LLMClassifier(
        thinking_budget=1000,
        max_tokens=2000  # Need enough tokens for thinking + output
    )
    clf.fit([], [])
    
    predictions = clf.predict([sample_jet], verbose=True)
    
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    assert clf.total_thinking_tokens > 0  # Should have used thinking


def test_llm_classifier_no_thinking(sample_jet):
    """Test prediction with thinking disabled."""
    clf = LLMClassifier(
        thinking_budget=0,  # Disable thinking
        max_tokens=100
    )
    clf.fit([], [])
    
    predictions = clf.predict([sample_jet], verbose=True)
    
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
    assert clf.total_thinking_tokens == 0  # Should not have used thinking


def test_llm_classifier_thinking_budget_variations(sample_jet):
    """Test that different thinking budgets produce different token usage."""
    budgets = [512, 2000, 5000]
    thinking_tokens = []
    
    for budget in budgets:
        clf = LLMClassifier(
            thinking_budget=budget,
            max_tokens=3000  # Need enough tokens for thinking + output
        )
        clf.fit([], [])
        clf.predict([sample_jet])
        thinking_tokens.append(clf.total_thinking_tokens)
    
    # At least one should be different (not all the same)
    assert len(set(thinking_tokens)) > 1, f"All budgets produced same tokens: {thinking_tokens}"


def test_llm_classifier_preview_prompt(sample_jet):
    """Test prompt preview functionality."""
    clf = LLMClassifier()
    clf.fit([], [])
    
    # Should not raise an error
    clf.preview_prompt(sample_jet)


def test_llm_classifier_parse_prediction():
    """Test prediction parsing."""
    clf = LLMClassifier()
    
    assert clf._parse_prediction("0") == 0
    assert clf._parse_prediction("1") == 1
    assert clf._parse_prediction("The answer is 0") == 0
    assert clf._parse_prediction("The answer is 1") == 1
    assert clf._parse_prediction("This is a quark jet") == 1
    assert clf._parse_prediction("This is a gluon jet") == 0


if __name__ == "__main__":
    # Run tests manually
    print("Running LLMClassifier tests...")
    
    # Load sample data
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'
    data = np.load(data_path)
    sample_jet = data['X'][0]
    
    # Test 1: Basic prediction with thinking
    print("\n" + "="*80)
    print("TEST 1: Basic prediction with thinking budget=1000")
    print("="*80)
    clf = LLMClassifier(thinking_budget=1000, max_tokens=2000)  # Need enough tokens for both thinking and output
    clf.fit([], [])
    pred = clf.predict([sample_jet], verbose=True)[0]
    print(f"Prediction: {pred}")
    
    # Test 2: Prediction without thinking
    print("\n" + "="*80)
    print("TEST 2: Prediction with thinking disabled (budget=0)")
    print("="*80)
    clf2 = LLMClassifier(thinking_budget=0, max_tokens=100)
    clf2.fit([], [])
    pred2 = clf2.predict([sample_jet], verbose=True)[0]
    print(f"Prediction: {pred2}")
    
    print("\n" + "="*80)
    print("All manual tests completed!")
    print("="*80)

