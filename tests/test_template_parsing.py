"""Tests for template parsing and feature inference."""

import pytest
from vibe_jet_tagging.utils.formatters import (
    format_features_as_text,
    infer_required_features,
    select_extractor_for_template,
)


def test_format_features_as_text_full():
    """Test formatting all features as text."""
    features = {
        'multiplicity': 42,
        'mean_pt': 12.345,
        'std_pt': 8.123,
        'median_pt': 9.876,
        'max_pt': 45.678,
        'lead_pt_frac': 0.234,
        'top3_pt_frac': 0.567,
        'top5_pt_frac': 0.789,
    }
    
    result = format_features_as_text(features)
    
    assert 'Jet Features:' in result
    assert 'Multiplicity: 42 particles' in result
    assert 'Mean pT: 12.35 GeV' in result
    assert 'pT Std Dev: 8.12 GeV' in result
    assert 'Median pT: 9.88 GeV' in result
    assert 'Max pT: 45.68 GeV' in result
    assert 'Leading pT Fraction: 0.234' in result
    assert 'Top-3 pT Fraction: 0.567' in result
    assert 'Top-5 pT Fraction: 0.789' in result


def test_format_features_as_text_partial():
    """Test formatting with only some features."""
    features = {
        'multiplicity': 30,
        'mean_pt': 15.5,
    }
    
    result = format_features_as_text(features)
    
    assert 'Multiplicity: 30 particles' in result
    assert 'Mean pT: 15.50 GeV' in result
    # Should not have other features
    assert 'Std Dev' not in result


def test_infer_required_features_generic():
    """Test inferring generic features placeholder."""
    template = "Here is the jet:\n{{jet_features}}\nClassify it."
    
    required = infer_required_features(template)
    
    assert 'features' in required


def test_infer_required_features_specific():
    """Test inferring specific feature placeholders."""
    template = "Multiplicity: {{multiplicity}}, Mean pT: {{mean_pt}}"
    
    required = infer_required_features(template)
    
    assert 'multiplicity' in required
    assert 'mean_pt' in required
    assert len(required) == 2


def test_infer_required_features_mixed():
    """Test inferring mixed placeholders."""
    template = """
    Jet data: {{jet_particles}}
    Features: {{jet_features}}
    Multiplicity: {{multiplicity}}
    """
    
    required = infer_required_features(template)
    
    # Should pick up jet_features and multiplicity
    # jet_particles is for raw data, not features
    assert 'features' in required
    assert 'multiplicity' in required


def test_infer_required_features_none():
    """Test template with no feature placeholders."""
    template = "Here is the jet:\n{{jet_particles}}\nClassify it."
    
    required = infer_required_features(template)
    
    # jet_particles is not a feature, so no features required
    assert len(required) == 0


def test_select_extractor_none():
    """Test extractor selection for template needing no features."""
    template = "Here is the jet:\n{{jet_particles}}\nClassify it."
    
    extractor = select_extractor_for_template(template)
    
    assert extractor == 'none'


def test_select_extractor_generic():
    """Test extractor selection for generic features."""
    template = "Features:\n{{jet_features}}\nClassify it."
    
    extractor = select_extractor_for_template(template)
    
    assert extractor == 'full'


def test_select_extractor_basic():
    """Test extractor selection for just multiplicity."""
    template = "This jet has {{multiplicity}} particles. Classify it."
    
    extractor = select_extractor_for_template(template)
    
    assert extractor == 'basic'


def test_select_extractor_kinematic():
    """Test extractor selection for kinematic features."""
    template = "Mean pT: {{mean_pt}}, Std: {{std_pt}}"
    
    extractor = select_extractor_for_template(template)
    
    assert extractor == 'kinematic'


def test_select_extractor_concentration():
    """Test extractor selection for concentration features."""
    template = "Leading fraction: {{lead_pt_frac}}"
    
    extractor = select_extractor_for_template(template)
    
    assert extractor == 'concentration'


def test_select_extractor_full_mixed():
    """Test extractor selection when mixing feature types."""
    template = """
    Multiplicity: {{multiplicity}}
    Mean pT: {{mean_pt}}
    Leading fraction: {{lead_pt_frac}}
    """
    
    extractor = select_extractor_for_template(template)
    
    # Needs both kinematic and concentration, so should be 'full'
    assert extractor == 'full'


def test_format_features_unknown():
    """Test formatting with unknown feature names."""
    features = {
        'multiplicity': 42,
        'unknown_feature': 123.456,
    }
    
    result = format_features_as_text(features)
    
    # Should handle known features normally
    assert 'Multiplicity: 42 particles' in result
    # Should have fallback for unknown
    assert 'unknown_feature' in result
    assert '123.456' in result


def test_infer_required_features_whitespace():
    """Test placeholder parsing with whitespace."""
    template = "Features: {{ jet_features }} and {{ multiplicity }}"
    
    required = infer_required_features(template)
    
    # Should handle whitespace correctly
    assert 'features' in required
    assert 'multiplicity' in required

