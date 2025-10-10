"""Tests for configuration system."""

import pytest
from vibe_jet_tagging.config import LLMConfig


def test_default_config():
    """Test default configuration creation."""
    config = LLMConfig()
    
    assert config.model_name == "gemini-2.5-flash-lite-preview-09-2025"
    assert config.template_name == "simple_list"
    assert config.format_type == "list"
    assert config.max_tokens == 1000
    assert config.thinking_budget is None
    assert config.batch_size == 1
    assert config.feature_extractor is None


def test_custom_config():
    """Test custom configuration creation."""
    config = LLMConfig(
        model_name="test-model",
        template_name="test_template",
        max_tokens=2000,
        thinking_budget=1500,
        batch_size=5,
        feature_extractor="full"
    )
    
    assert config.model_name == "test-model"
    assert config.template_name == "test_template"
    assert config.max_tokens == 2000
    assert config.thinking_budget == 1500
    assert config.batch_size == 5
    assert config.feature_extractor == "full"


def test_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        'model_name': 'test-model',
        'template_name': 'test_template',
        'max_tokens': 1500,
        'thinking_budget': 1000,
    }
    
    config = LLMConfig.from_dict(config_dict)
    
    assert config.model_name == 'test-model'
    assert config.template_name == 'test_template'
    assert config.max_tokens == 1500
    assert config.thinking_budget == 1000


def test_to_dict():
    """Test converting config to dictionary."""
    config = LLMConfig(model_name='test-model', max_tokens=1500)
    config_dict = config.to_dict()
    
    assert config_dict['model_name'] == 'test-model'
    assert config_dict['max_tokens'] == 1500
    assert 'template_name' in config_dict


def test_validation_max_tokens():
    """Test validation of max_tokens."""
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        LLMConfig(max_tokens=0)
    
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        LLMConfig(max_tokens=-100)


def test_validation_thinking_budget():
    """Test validation of thinking_budget."""
    with pytest.raises(ValueError, match="thinking_budget must be non-negative"):
        LLMConfig(thinking_budget=-1)
    
    # Valid values
    config = LLMConfig(thinking_budget=0)
    assert config.thinking_budget == 0
    
    config = LLMConfig(thinking_budget=1000)
    assert config.thinking_budget == 1000


def test_validation_format_type():
    """Test validation of format_type."""
    with pytest.raises(ValueError, match="format_type must be"):
        LLMConfig(format_type='invalid')
    
    # Valid values
    for fmt in ['list', 'yaml', 'table']:
        config = LLMConfig(format_type=fmt)
        assert config.format_type == fmt


def test_validation_batch_size():
    """Test validation of batch_size."""
    with pytest.raises(ValueError, match="batch_size must be positive"):
        LLMConfig(batch_size=0)
    
    with pytest.raises(ValueError, match="batch_size must be positive"):
        LLMConfig(batch_size=-5)


def test_validation_feature_extractor():
    """Test validation of feature_extractor."""
    with pytest.raises(ValueError, match="feature_extractor must be one of"):
        LLMConfig(feature_extractor='invalid')
    
    # Valid values
    for extractor in ['basic', 'kinematic', 'concentration', 'full', 'none']:
        config = LLMConfig(feature_extractor=extractor)
        assert config.feature_extractor == extractor
    
    # None is also valid
    config = LLMConfig(feature_extractor=None)
    assert config.feature_extractor is None

