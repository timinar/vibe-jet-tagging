"""LLM-based classifier for jet tagging using OpenRouter API."""

import os
import re
from typing import Any, Optional

from openai import OpenAI

from vibe_jet_tagging.classifier import Classifier
from vibe_jet_tagging.utils.formatters import fill_template, load_template


class LLMClassifier(Classifier):
    """
    Classifier that uses LLMs via OpenRouter for zero-shot jet classification.
    
    Parameters
    ----------
    model_name : str
        Model identifier for OpenRouter (e.g., "anthropic/claude-3.5-sonnet")
    template_name : str
        Name of the prompt template to use
    format_type : str
        Data format type: 'list', 'yaml', or 'table'
    api_key : str, optional
        OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
    templates_dir : str
        Directory containing prompt templates
    base_url : str
        OpenRouter API base URL
    """
    
    def __init__(
        self,
        model_name: str = "anthropic/claude-3.5-sonnet",
        template_name: str = "simple_list",
        format_type: str = "list",
        api_key: Optional[str] = None,
        templates_dir: str = "templates",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model_name = model_name
        self.template_name = template_name
        self.format_type = format_type
        self.templates_dir = templates_dir
        
        # Load template
        self.template = load_template(template_name, templates_dir)
        
        # Set up OpenAI client with OpenRouter
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key parameter."
            )
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        
        self.is_fitted = False
    
    def fit(self, X: list[Any], y: list[Any]) -> "LLMClassifier":
        """
        Prepare classifier (no-op for zero-shot).
        
        For zero-shot classification, no training is needed.
        In future, this could store few-shot examples.
        
        Parameters
        ----------
        X : list
            Training jets (not used in zero-shot)
        y : list
            Training labels (not used in zero-shot)
        
        Returns
        -------
        LLMClassifier
            Returns self
        """
        self.is_fitted = True
        return self
    
    def predict(self, X: list[Any]) -> list[int]:
        """
        Predict jet labels using LLM.
        
        Parameters
        ----------
        X : list
            List of jets to classify
        
        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        predictions = []
        for jet in X:
            prediction = self._predict_single(jet)
            predictions.append(prediction)
        
        return predictions
    
    def _predict_single(self, jet: Any) -> int:
        """
        Predict label for a single jet.
        
        Parameters
        ----------
        jet : array-like
            Single jet data
        
        Returns
        -------
        int
            Predicted label (0 or 1)
        """
        # Format jet and fill template
        prompt = fill_template(self.template, jet, self.format_type)
        
        # Call OpenRouter API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            
            # Extract response
            content = response.choices[0].message.content.strip()
            
            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction
            
        except Exception as e:
            print(f"Error calling API: {e}")
            # Return random guess on error
            import random
            return random.randint(0, 1)
    
    def _parse_prediction(self, response: str) -> int:
        """
        Parse LLM response to extract 0 or 1.
        
        Parameters
        ----------
        response : str
            LLM response text
        
        Returns
        -------
        int
            Parsed label (0 or 1)
        """
        # Look for 0 or 1 in the response
        match = re.search(r'\b[01]\b', response)
        if match:
            return int(match.group())
        
        # If not found, check for keywords
        response_lower = response.lower()
        if "quark" in response_lower or "1" in response:
            return 1
        elif "gluon" in response_lower or "0" in response:
            return 0
        
        # Default to 0 if unclear
        print(f"Warning: Could not parse prediction from response: '{response}'. Defaulting to 0.")
        return 0

