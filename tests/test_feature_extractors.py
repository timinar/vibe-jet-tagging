"""Tests for feature extractors."""

import numpy as np
import pytest
from vibe_jet_tagging.feature_extractors import (
    BasicExtractor,
    KinematicExtractor,
    ConcentrationExtractor,
    FullExtractor,
    get_extractor,
)


@pytest.fixture
def sample_jet():
    """Create a sample jet with known properties."""
    # Jet with 5 particles: pt values [10, 5, 3, 2, 1]
    jet = np.array([
        [10.0, 0.5, 1.0, 211],   # Particle 1
        [5.0, -0.3, 2.0, -211],  # Particle 2
        [3.0, 0.1, 1.5, 22],     # Particle 3
        [2.0, 0.0, 1.8, 211],    # Particle 4
        [1.0, 0.2, 1.2, 22],     # Particle 5
        [0.0, 0.0, 0.0, 0],      # Padding
        [0.0, 0.0, 0.0, 0],      # Padding
    ], dtype=np.float32)
    return jet


@pytest.fixture
def empty_jet():
    """Create an empty jet (all padding)."""
    jet = np.zeros((10, 4), dtype=np.float32)
    return jet


def test_basic_extractor(sample_jet):
    """Test BasicExtractor on sample jet."""
    extractor = BasicExtractor()
    features = extractor.extract(sample_jet)
    
    assert 'multiplicity' in features
    assert features['multiplicity'] == 5
    assert extractor.feature_names == ['multiplicity']


def test_basic_extractor_empty(empty_jet):
    """Test BasicExtractor on empty jet."""
    extractor = BasicExtractor()
    features = extractor.extract(empty_jet)
    
    assert features['multiplicity'] == 0


def test_kinematic_extractor(sample_jet):
    """Test KinematicExtractor on sample jet."""
    extractor = KinematicExtractor()
    features = extractor.extract(sample_jet)
    
    # pt values: [10, 5, 3, 2, 1]
    # mean = 21/5 = 4.2
    # max = 10
    # median = 3
    
    assert 'multiplicity' in features
    assert 'mean_pt' in features
    assert 'std_pt' in features
    assert 'median_pt' in features
    assert 'max_pt' in features
    
    assert features['multiplicity'] == 5
    assert abs(features['mean_pt'] - 4.2) < 0.01
    assert features['max_pt'] == 10.0
    assert features['median_pt'] == 3.0
    
    assert len(extractor.feature_names) == 5


def test_kinematic_extractor_empty(empty_jet):
    """Test KinematicExtractor on empty jet."""
    extractor = KinematicExtractor()
    features = extractor.extract(empty_jet)
    
    assert features['multiplicity'] == 0
    assert features['mean_pt'] == 0.0
    assert features['std_pt'] == 0.0
    assert features['median_pt'] == 0.0
    assert features['max_pt'] == 0.0


def test_concentration_extractor(sample_jet):
    """Test ConcentrationExtractor on sample jet."""
    extractor = ConcentrationExtractor()
    features = extractor.extract(sample_jet)
    
    # pt values: [10, 5, 3, 2, 1], sum = 21
    # lead_pt_frac = 10/21 ≈ 0.476
    # top3_pt_frac = (10+5+3)/21 = 18/21 ≈ 0.857
    # top5_pt_frac = 21/21 = 1.0
    
    assert 'lead_pt_frac' in features
    assert 'top3_pt_frac' in features
    assert 'top5_pt_frac' in features
    
    assert abs(features['lead_pt_frac'] - 10/21) < 0.01
    assert abs(features['top3_pt_frac'] - 18/21) < 0.01
    assert abs(features['top5_pt_frac'] - 1.0) < 0.01
    
    assert len(extractor.feature_names) == 3


def test_concentration_extractor_empty(empty_jet):
    """Test ConcentrationExtractor on empty jet."""
    extractor = ConcentrationExtractor()
    features = extractor.extract(empty_jet)
    
    assert features['lead_pt_frac'] == 0.0
    assert features['top3_pt_frac'] == 0.0
    assert features['top5_pt_frac'] == 0.0


def test_full_extractor(sample_jet):
    """Test FullExtractor combines all features."""
    extractor = FullExtractor()
    features = extractor.extract(sample_jet)
    
    # Should have all features from kinematic and concentration
    assert 'multiplicity' in features
    assert 'mean_pt' in features
    assert 'std_pt' in features
    assert 'median_pt' in features
    assert 'max_pt' in features
    assert 'lead_pt_frac' in features
    assert 'top3_pt_frac' in features
    assert 'top5_pt_frac' in features
    
    # Check values match
    assert features['multiplicity'] == 5
    assert abs(features['mean_pt'] - 4.2) < 0.01
    assert abs(features['lead_pt_frac'] - 10/21) < 0.01
    
    assert len(extractor.feature_names) == 8


def test_batch_extraction(sample_jet):
    """Test batch extraction."""
    extractor = BasicExtractor()
    
    # Create batch of 3 jets (all same for simplicity)
    jets = np.stack([sample_jet, sample_jet, sample_jet])
    
    features_list = extractor.extract_batch(jets)
    
    assert len(features_list) == 3
    assert all(f['multiplicity'] == 5 for f in features_list)


def test_get_extractor():
    """Test get_extractor factory function."""
    basic = get_extractor('basic')
    assert isinstance(basic, BasicExtractor)
    
    kinematic = get_extractor('kinematic')
    assert isinstance(kinematic, KinematicExtractor)
    
    concentration = get_extractor('concentration')
    assert isinstance(concentration, ConcentrationExtractor)
    
    full = get_extractor('full')
    assert isinstance(full, FullExtractor)
    
    with pytest.raises(ValueError, match="Unknown extractor"):
        get_extractor('invalid')


def test_single_particle_jet():
    """Test extractors on jet with single particle."""
    jet = np.array([
        [10.0, 0.5, 1.0, 211],
        [0.0, 0.0, 0.0, 0],
    ], dtype=np.float32)
    
    extractor = FullExtractor()
    features = extractor.extract(jet)
    
    assert features['multiplicity'] == 1
    assert features['mean_pt'] == 10.0
    assert features['std_pt'] == 0.0  # Only one particle
    assert features['median_pt'] == 10.0
    assert features['max_pt'] == 10.0
    assert features['lead_pt_frac'] == 1.0  # All pt in one particle
    assert features['top3_pt_frac'] == 1.0
    assert features['top5_pt_frac'] == 1.0

