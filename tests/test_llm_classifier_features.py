"""Tests for LLMClassifier with feature extraction."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from vibe_jet_tagging import LLMClassifier
from vibe_jet_tagging.config import LLMConfig
from vibe_jet_tagging.feature_extractors import BasicExtractor, FullExtractor


@pytest.fixture
def sample_jet():
    """Create a sample jet for testing."""
    jet = np.array([
        [10.0, 0.5, 1.0, 211],
        [5.0, -0.3, 2.0, -211],
        [3.0, 0.1, 1.5, 22],
        [0.0, 0.0, 0.0, 0],
    ], dtype=np.float32)
    return jet


@pytest.fixture
def mock_api_key(monkeypatch):
    """Mock the API key environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")


def test_backward_compatibility_init(mock_api_key):
    """Test that old-style initialization still works."""
    clf = LLMClassifier(
        model_name="test-model",
        template_name="simple_list",
        max_tokens=500,
    )
    
    assert clf.model_name == "test-model"
    assert clf.template_name == "simple_list"
    assert clf.max_tokens == 500
    # Should have auto-detected no features needed for simple_list
    assert clf.feature_extractor is None


def test_config_based_init(mock_api_key):
    """Test config-based initialization."""
    config = LLMConfig(
        model_name="test-model",
        template_name="simple_list",
        max_tokens=500,
        feature_extractor="basic",
    )
    
    clf = LLMClassifier(config=config)
    
    assert clf.model_name == "test-model"
    assert clf.max_tokens == 500
    assert isinstance(clf.feature_extractor, BasicExtractor)


def test_config_dict_init(mock_api_key):
    """Test initialization with config dictionary."""
    config_dict = {
        'model_name': 'test-model',
        'template_name': 'simple_list',
        'max_tokens': 500,
        'feature_extractor': 'full',
    }
    
    clf = LLMClassifier(config=config_dict)
    
    assert clf.model_name == "test-model"
    assert isinstance(clf.feature_extractor, FullExtractor)


def test_explicit_feature_extractor(mock_api_key):
    """Test explicit feature extractor specification."""
    clf = LLMClassifier(
        template_name="simple_list",
        feature_extractor="kinematic",
    )
    
    assert clf.feature_extractor is not None
    assert 'mean_pt' in clf.feature_extractor.feature_names


def test_feature_extractor_none(mock_api_key):
    """Test explicitly disabling feature extraction."""
    clf = LLMClassifier(
        template_name="simple_list",
        feature_extractor="none",
    )
    
    assert clf.feature_extractor is None


def test_build_prompt_without_features(mock_api_key, sample_jet):
    """Test prompt building without feature extraction."""
    clf = LLMClassifier(
        template_name="simple_list",
        feature_extractor="none",
    )
    
    clf.fit([], [])
    prompt = clf._build_prompt(sample_jet)
    
    # Should contain raw particle data
    assert "Particle" in prompt
    assert "pt=" in prompt
    # Should not contain feature text
    assert "Jet Features" not in prompt


def test_build_prompt_with_features(mock_api_key, sample_jet):
    """Test prompt building with feature extraction."""
    # Create a simple template that uses features
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir) / "test_features.txt"
        template_path.write_text(
            "Classify this jet:\n{{jet_features}}\nIs it quark (1) or gluon (0)?"
        )
        
        clf = LLMClassifier(
            template_name="test_features",
            templates_dir=tmpdir,
            feature_extractor="basic",
        )
        
        clf.fit([], [])
        prompt = clf._build_prompt(sample_jet)
        
        # Should contain feature text
        assert "Jet Features" in prompt
        assert "Multiplicity: 3 particles" in prompt
        # Should not contain raw particles
        assert "Particle 1:" not in prompt


def test_preview_prompt_shows_extractor(mock_api_key, sample_jet, capsys):
    """Test that preview_prompt shows feature extractor info."""
    clf = LLMClassifier(
        template_name="simple_list",
        feature_extractor="basic",
    )
    
    clf.fit([], [])
    clf.preview_prompt(sample_jet)
    
    captured = capsys.readouterr()
    assert "Feature Extractor: BasicExtractor" in captured.out


def test_auto_detect_feature_extractor_from_template(mock_api_key):
    """Test that feature extractor is auto-detected from template."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Template that requires features
        template_path = Path(tmpdir) / "auto_features.txt"
        template_path.write_text(
            "This jet has {{multiplicity}} particles with mean pt {{mean_pt}}."
        )
        
        clf = LLMClassifier(
            template_name="auto_features",
            templates_dir=tmpdir,
            # Don't specify feature_extractor - should auto-detect
        )
        
        # Should have auto-detected kinematic extractor
        # (needs multiplicity and mean_pt)
        assert clf.feature_extractor is not None
        assert 'multiplicity' in clf.feature_extractor.feature_names
        assert 'mean_pt' in clf.feature_extractor.feature_names


def test_individual_feature_placeholders(mock_api_key, sample_jet):
    """Test that individual feature placeholders are replaced."""
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir) / "individual_features.txt"
        template_path.write_text(
            "Jet has {{multiplicity}} particles, mean pt = {{mean_pt}} GeV"
        )
        
        clf = LLMClassifier(
            template_name="individual_features",
            templates_dir=tmpdir,
        )
        
        clf.fit([], [])
        prompt = clf._build_prompt(sample_jet)
        
        # Should have replaced placeholders
        assert "{{multiplicity}}" not in prompt
        assert "{{mean_pt}}" not in prompt
        assert "3 particles" in prompt
        assert "GeV" in prompt


def test_batch_size_stored(mock_api_key):
    """Test that batch_size is stored correctly."""
    clf = LLMClassifier(batch_size=10)
    assert clf.batch_size == 10
    
    config = LLMConfig(batch_size=5)
    clf2 = LLMClassifier(config=config)
    assert clf2.batch_size == 5

