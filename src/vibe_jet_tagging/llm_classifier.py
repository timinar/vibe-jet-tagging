"""LLM-based classifier for jet tagging using Google Gemini API."""

import asyncio
import os
import re
from typing import Any, Optional, Union

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm
from google import genai
from google.genai import types

from vibe_jet_tagging.classifier import Classifier
from vibe_jet_tagging.config import LLMConfig
from vibe_jet_tagging.feature_extractors import FeatureExtractor, get_extractor
from vibe_jet_tagging.utils.formatters import (
    fill_template,
    load_template,
    format_features_as_text,
    select_extractor_for_template,
)


class LLMClassifier(Classifier):
    """
    Classifier that uses Google Gemini for zero-shot jet classification.
    
    Supports both traditional parameter-based initialization and config-based initialization.
    Can optionally use feature extraction to provide high-level jet features to the LLM
    instead of raw particle data.
    
    Parameters
    ----------
    config : Union[LLMConfig, dict], optional
        Configuration object or dictionary. If provided, individual parameters are ignored.
    model_name : str
        Gemini model identifier (e.g., "gemini-2.5-flash-lite-preview-09-2025")
    template_name : str
        Name of the prompt template to use
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
        Number of jets to process in parallel (not yet implemented)
    feature_extractor : Union[str, FeatureExtractor], optional
        Feature extractor to use. Can be:
        - String: 'basic', 'kinematic', 'concentration', 'full', or 'none'
        - FeatureExtractor instance
        - None: auto-detect from template requirements
    """
    
    def __init__(
        self,
        config: Optional[Union[LLMConfig, dict]] = None,
        model_name: str = "gemini-2.5-flash-lite-preview-09-2025",
        template_name: str = "simple_list",
        format_type: str = "list",
        max_tokens: int = 1000,
        thinking_budget: Optional[int] = None,
        api_key: Optional[str] = None,
        templates_dir: str = "templates",
        batch_size: int = 1,
        feature_extractor: Optional[Union[str, FeatureExtractor]] = None,
    ):
        # Handle config-based initialization
        if config is not None:
            if isinstance(config, dict):
                config = LLMConfig.from_dict(config)
            self.config = config
            model_name = config.model_name
            template_name = config.template_name
            format_type = config.format_type
            max_tokens = config.max_tokens
            thinking_budget = config.thinking_budget
            api_key = config.api_key
            templates_dir = config.templates_dir
            batch_size = config.batch_size
            if feature_extractor is None:
                feature_extractor = config.feature_extractor
        else:
            # Create config from individual parameters
            self.config = LLMConfig(
                model_name=model_name,
                template_name=template_name,
                format_type=format_type,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                api_key=api_key,
                templates_dir=templates_dir,
                batch_size=batch_size,
                feature_extractor=feature_extractor if isinstance(feature_extractor, str) else None,
            )
        
        self.model_name = model_name
        self.template_name = template_name
        self.format_type = format_type
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.templates_dir = templates_dir
        self.batch_size = batch_size
        
        # Load template
        self.template = load_template(template_name, templates_dir)
        
        # Set up feature extractor
        self._setup_feature_extractor(feature_extractor)
        
        # Set up Gemini client
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=api_key)
        
        self.is_fitted = False
        
        # Track cumulative token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_thinking_tokens = 0
        self.total_cost = 0.0
    
    def _setup_feature_extractor(self, feature_extractor: Optional[Union[str, FeatureExtractor]]):
        """
        Set up the feature extractor based on configuration or auto-detection.
        
        Parameters
        ----------
        feature_extractor : Union[str, FeatureExtractor], optional
            Feature extractor specification
        """
        if isinstance(feature_extractor, FeatureExtractor):
            # Already an instance
            self.feature_extractor = feature_extractor
        elif isinstance(feature_extractor, str):
            if feature_extractor == 'none':
                self.feature_extractor = None
            else:
                self.feature_extractor = get_extractor(feature_extractor)
        elif feature_extractor is None:
            # Auto-detect from template
            extractor_name = select_extractor_for_template(self.template)
            if extractor_name == 'none':
                self.feature_extractor = None
            else:
                self.feature_extractor = get_extractor(extractor_name)
        else:
            raise ValueError(
                f"feature_extractor must be str, FeatureExtractor, or None, got {type(feature_extractor)}"
            )
    
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
    
    def _build_prompt(self, jet: Any) -> str:
        """
        Build prompt for a single jet, including features if extractor is configured.
        
        Parameters
        ----------
        jet : array-like
            Single jet data
        
        Returns
        -------
        str
            Formatted prompt
        """
        # Start with basic template filling
        prompt = fill_template(self.template, jet, self.format_type)
        
        # Add features if extractor is configured
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract(jet)
            features_text = format_features_as_text(features)
            
            # Replace {{jet_features}} placeholder
            prompt = prompt.replace("{{jet_features}}", features_text)
            
            # Also replace individual feature placeholders
            for key, value in features.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in prompt:
                    if key == 'multiplicity':
                        prompt = prompt.replace(placeholder, f"{int(value)}")
                    else:
                        prompt = prompt.replace(placeholder, f"{value:.3f}")
        
        return prompt
    
    def preview_prompt(self, jet: Any) -> str:
        """
        Preview the prompt that will be sent to Gemini for a given jet.
        
        Parameters
        ----------
        jet : array-like
            Single jet data to preview
        
        Returns
        -------
        str
            The formatted prompt
        """
        prompt = self._build_prompt(jet)
        
        print("=" * 80)
        print("PROMPT PREVIEW")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Template: {self.template_name}")
        print(f"Format: {self.format_type}")
        print(f"Feature Extractor: {type(self.feature_extractor).__name__ if self.feature_extractor else 'None'}")
        print(f"Max output tokens: {self.max_tokens}")
        print(f"Thinking budget: {self.thinking_budget}")
        print("\n" + "-" * 80)
        print("PROMPT:")
        print("-" * 80)
        print(prompt)
        print("=" * 80)
        
        return prompt
    
    def predict(self, X: list[Any], verbose: bool = False, batch_size: Optional[int] = None) -> list[int]:
        """
        Predict jet labels using Gemini.
        
        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and cost information
        batch_size : int, optional
            If specified, use async batching with this batch size for parallel requests.
            If None (default), process sequentially.
        
        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        if batch_size is not None:
            # Use async batching
            return asyncio.run(self.predict_async(X, verbose=verbose, batch_size=batch_size))
        
        # Sequential processing (original behavior)
        predictions = []
        for jet in tqdm(X):
            prediction = self._predict_single(jet, verbose=verbose)
            predictions.append(prediction)
        
        if verbose:
            self._print_cumulative_stats()
        
        return predictions
    
    async def predict_async(self, X: list[Any], verbose: bool = False, batch_size: int = 10) -> list[int]:
        """
        Predict jet labels using Gemini with async batching.
        
        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and cost information
        batch_size : int
            Number of concurrent requests to process in parallel
        
        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")
        
        # Process in batches
        predictions = []
        total = len(X)
        
        with tqdm(total=total, desc="Predicting (async)") as pbar:
            for i in range(0, total, batch_size):
                batch = X[i:i + batch_size]
                batch_predictions = await asyncio.gather(
                    *[self._predict_single_async(jet, verbose=verbose) for jet in batch]
                )
                predictions.extend(batch_predictions)
                pbar.update(len(batch))
        
        if verbose:
            self._print_cumulative_stats()
        
        return predictions
    
    async def _predict_single_async(self, jet: Any, verbose: bool = False) -> int:
        """
        Async version of _predict_single for concurrent requests.
        
        Parameters
        ----------
        jet : array-like
            Single jet data
        verbose : bool
            If True, print detailed token usage and cost information
        
        Returns
        -------
        int
            Predicted label (0 or 1)
        """
        # Build prompt with features if extractor is configured
        prompt = self._build_prompt(jet)
        
        # Call Gemini API asynchronously
        try:
            # Build generation config
            config_params = {
                "max_output_tokens": self.max_tokens
            }
            
            # Add thinking config if specified
            if self.thinking_budget is not None:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            
            generation_config = types.GenerateContentConfig(**config_params)
            
            # Use async API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=generation_config
                )
            )
            
            # Extract response
            content = response.text if response.text else ""
            content = content.strip()
            
            # Track usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                prompt_tokens = usage.prompt_token_count
                completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
                
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_thinking_tokens += thinking_tokens
                
                # Calculate cost
                input_cost_per_token = 0.000000075
                output_cost_per_token = 0.0000003
                input_cost = prompt_tokens * input_cost_per_token
                output_cost = (completion_tokens + thinking_tokens) * output_cost_per_token
                call_cost = input_cost + output_cost
                self.total_cost += call_cost
            
            # Extract label (expecting 0 or 1)
            match = re.search(r'\b[01]\b', content)
            if match:
                return int(match.group())
            else:
                print(f"Warning: Could not parse response '{content[:100]}', defaulting to 0")
                return 0
                
        except Exception as e:
            print(f"Error in async prediction: {e}")
            return 0
    
    def _predict_single(self, jet: Any, verbose: bool = False) -> int:
        """
        Predict label for a single jet.
        
        Parameters
        ----------
        jet : array-like
            Single jet data
        verbose : bool
            If True, print detailed token usage and cost information
        
        Returns
        -------
        int
            Predicted label (0 or 1)
        """
        # Build prompt with features if extractor is configured
        prompt = self._build_prompt(jet)
        
        # Call Gemini API
        try:
            # Build generation config
            config_params = {
                "max_output_tokens": self.max_tokens
            }
            
            # Add thinking config if specified
            if self.thinking_budget is not None:
                config_params["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
            
            if verbose:
                print(f"\n🔧 API PARAMETERS")
                print(f"Model: {self.model_name}")
                print(f"Max output tokens: {self.max_tokens}")
                print(f"Thinking budget: {self.thinking_budget}")
                print()
            
            generation_config = types.GenerateContentConfig(**config_params)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config
            )
            
            # Extract response
            content = response.text if response.text else ""
            content = content.strip()
            
            if len(content) == 0:
                print("Warning: API returned empty content")
                print(f"Response object: {response}")
                return 0
            
            # Track token usage
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            completion_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0
            total_tokens = usage.total_token_count
            
            # Update cumulative totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_thinking_tokens += thinking_tokens
            
            # Calculate cost (Gemini Flash-Lite pricing as example)
            # Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
            input_cost_per_token = 0.000000075
            output_cost_per_token = 0.0000003
            
            input_cost = prompt_tokens * input_cost_per_token
            output_cost = (completion_tokens + thinking_tokens) * output_cost_per_token
            call_cost = input_cost + output_cost
            
            self.total_cost += call_cost
            
            if verbose:
                print("\n" + "─" * 60)
                print("📊 TOKEN USAGE")
                print("─" * 60)
                print(f"Prompt tokens:     {prompt_tokens:,}")
                print(f"Completion tokens: {completion_tokens:,}")
                if thinking_tokens > 0:
                    print(f"Thinking tokens:   {thinking_tokens:,}")
                    print(f"├─ Thinking:       {thinking_tokens:,}")
                    print(f"└─ Output:         {completion_tokens:,}")
                print(f"Total tokens:      {total_tokens:,}")
                
                print(f"\n💰 COST")
                print(f"Input cost:        ${input_cost:.6f}")
                print(f"Output cost:       ${output_cost:.6f}")
                print(f"Call cost:         ${call_cost:.6f}")
                
                # Show final response
                print(f"\n✨ RESPONSE")
                print("─" * 60)
                print(f"Content: {content}")
                print("─" * 60 + "\n")
            
            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Return random guess on error
            import random
            return random.randint(0, 1)
    
    def _parse_prediction(self, response: str) -> int:
        """
        Parse Gemini response to extract 0 or 1.
        
        Parameters
        ----------
        response : str
            Gemini response text
        
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
        print(f"Warning: Could not parse prediction from response: '{response}'")
        return 0
    
    def _print_cumulative_stats(self) -> None:
        """Print cumulative token usage and cost statistics."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens + self.total_thinking_tokens
        
        print("\n" + "═" * 60)
        print("📈 CUMULATIVE STATISTICS")
        print("═" * 60)
        print(f"Total prompt tokens:     {self.total_prompt_tokens:,}")
        print(f"Total completion tokens: {self.total_completion_tokens:,}")
        if self.total_thinking_tokens > 0:
            print(f"Total thinking tokens:   {self.total_thinking_tokens:,}")
        print(f"Total tokens:            {total_tokens:,}")
        print(f"\n💰 Total estimated cost: ${self.total_cost:.6f}")
        print("═" * 60 + "\n")
