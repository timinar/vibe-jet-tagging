"""Configuration management for LLMClassifier."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """
    Configuration for LLMClassifier.
    
    Parameters
    ----------
    model_name : str
        Gemini model identifier (e.g., "gemini-2.5-flash-lite-preview-09-2025")
    template_name : str
        Name of the prompt template to use (without .txt extension)
    format_type : str
        Data format type: 'list', 'yaml', or 'table'
    max_tokens : int
        Maximum output tokens for the response
    thinking_budget : int, optional
        Thinking budget for reasoning (in tokens)
    api_key : str, optional
        Gemini API key. If None, reads from GEMINI_API_KEY env var
    templates_dir : str
        Directory containing prompt templates
    batch_size : int
        Number of jets to process in parallel (default=1)
    feature_extractor : str, optional
        Name of feature extractor to use: 'basic', 'kinematic', 'concentration', 'full'
        If None, auto-inferred from template requirements
    """
    
    model_name: str = "gemini-2.5-flash-lite-preview-09-2025"
    template_name: str = "simple_list"
    format_type: str = "list"
    max_tokens: int = 1000
    thinking_budget: Optional[int] = None
    api_key: Optional[str] = None
    templates_dir: str = "templates"
    batch_size: int = 1
    feature_extractor: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        
        if self.thinking_budget is not None and self.thinking_budget < 0:
            raise ValueError(f"thinking_budget must be non-negative, got {self.thinking_budget}")
        
        if self.format_type not in ['list', 'yaml', 'table']:
            raise ValueError(f"format_type must be 'list', 'yaml', or 'table', got {self.format_type}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.feature_extractor is not None:
            valid_extractors = ['basic', 'kinematic', 'concentration', 'full', 'none']
            if self.feature_extractor not in valid_extractors:
                raise ValueError(
                    f"feature_extractor must be one of {valid_extractors}, got {self.feature_extractor}"
                )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "LLMConfig":
        """
        Create LLMConfig from dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary
        
        Returns
        -------
        LLMConfig
            Configuration instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        
        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            'model_name': self.model_name,
            'template_name': self.template_name,
            'format_type': self.format_type,
            'max_tokens': self.max_tokens,
            'thinking_budget': self.thinking_budget,
            'api_key': self.api_key,
            'templates_dir': self.templates_dir,
            'batch_size': self.batch_size,
            'feature_extractor': self.feature_extractor,
        }

