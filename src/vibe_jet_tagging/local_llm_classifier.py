"""LLM-based classifier for jet tagging using local OpenAI-compatible API."""

import asyncio
import re
from typing import Any, Literal, Optional

from openai import AsyncOpenAI, OpenAI

from vibe_jet_tagging.classifier import Classifier
from vibe_jet_tagging.utils.formatters import fill_template, load_template


class LocalLLMClassifier(Classifier):
    """
    Classifier that uses local OpenAI-compatible API for zero-shot jet classification.

    Uses reasoning models with configurable effort levels for controlling reasoning depth.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., "openai/gpt-oss-120b")
    template_name : str
        Name of the prompt template to use
    format_type : str
        Data format type: 'list', 'yaml', or 'table'
    reasoning_effort : str
        Reasoning effort level: 'low', 'medium', or 'high'
        - 'low': Fast reasoning with minimal depth
        - 'medium': Balanced reasoning (default)
        - 'high': Deep reasoning with maximum effort
    reasoning_summary : str
        Controls reasoning summary detail: 'auto', 'concise', or 'detailed'
    base_url : str
        Base URL for the OpenAI-compatible API (default: "http://localhost:8000/v1")
    api_key : str
        API key (use "EMPTY" for local servers without authentication)
    templates_dir : str
        Directory containing prompt templates
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        template_name: str = "simple_list",
        format_type: str = "list",
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        reasoning_summary: Literal["auto", "concise", "detailed"] = "auto",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        templates_dir: str = "templates",
    ):
        self.model_name = model_name
        self.template_name = template_name
        self.format_type = format_type
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.templates_dir = templates_dir

        # Load template
        self.template = load_template(template_name, templates_dir)

        # Set up both sync and async OpenAI clients with local server
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.is_fitted = False

        # Track cumulative token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost = 0.0

    def fit(self, X: list[Any], y: list[Any]) -> "LocalLLMClassifier":
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
        LocalLLMClassifier
            Returns self
        """
        self.is_fitted = True
        return self

    def preview_prompt(self, jet: Any) -> None:
        """
        Preview the prompt that will be sent to the model for a given jet.

        Parameters
        ----------
        jet : array-like
            Single jet data to preview
        """
        prompt = fill_template(self.template, jet, self.format_type)

        print("=" * 80)
        print("PROMPT PREVIEW")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Template: {self.template_name}")
        print(f"Format: {self.format_type}")
        print(f"Reasoning effort: {self.reasoning_effort}")
        print(f"Reasoning summary: {self.reasoning_summary}")
        print("\n" + "-" * 80)
        print("PROMPT:")
        print("-" * 80)
        print(prompt)
        print("=" * 80)

    def predict(self, X: list[Any], verbose: bool = False, use_async: bool = True) -> list[int]:
        """
        Predict jet labels using local LLM.

        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and cost information
        use_async : bool
            If True, use async/concurrent requests (default: True)
            Set to False for sequential processing

        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")

        if use_async:
            # Use async for concurrent requests
            predictions = asyncio.run(self._predict_async(X, verbose=verbose))
        else:
            # Sequential processing
            predictions = []
            for jet in X:
                prediction = self._predict_single(jet, verbose=verbose)
                predictions.append(prediction)

        if verbose:
            self._print_cumulative_stats()

        return predictions

    async def _predict_async(self, X: list[Any], verbose: bool = False) -> list[int]:
        """
        Predict labels for multiple jets concurrently using async.

        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and cost information

        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        # Create concurrent tasks for all jets
        tasks = [self._predict_single_async(jet, verbose=verbose) for jet in X]

        # Run all tasks concurrently
        predictions = await asyncio.gather(*tasks)

        return list(predictions)

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
        # Format jet and fill template
        prompt = fill_template(self.template, jet, self.format_type)

        # Call local LLM API
        try:
            if verbose:
                print(f"\n🔧 API PARAMETERS")
                print(f"Model: {self.model_name}")
                print(f"Reasoning effort: {self.reasoning_effort}")
                print(f"Reasoning summary: {self.reasoning_summary}")
                print()

            response = self.client.responses.create(
                model=self.model_name,
                instructions="You are a helpful assistant.",
                input=prompt,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary
                }
            )

            # Extract response content
            # The response.output is a list of ResponseReasoningItem objects
            # For reasoning models, the final answer is at the end of the reasoning text
            content = ""
            if hasattr(response, 'output') and isinstance(response.output, list):
                # Collect all reasoning text - the answer is typically at the end
                all_text = []
                for item in response.output:
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                all_text.append(content_item.text)

                # Combine all text - the final answer should be in there
                content = " ".join(all_text)

            # Convert to string and strip
            content = str(content).strip()

            if len(content) == 0:
                print("Warning: API returned empty content")
                print(f"Response object: {response}")
                return 0

            # Track token usage if available
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
                total_tokens = getattr(usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens + reasoning_tokens)

                # Update cumulative totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_reasoning_tokens += reasoning_tokens

                # Calculate cost (example pricing - adjust based on actual costs)
                # Using similar rates to Gemini for consistency
                input_cost_per_token = 0.000000075
                output_cost_per_token = 0.0000003

                input_cost = prompt_tokens * input_cost_per_token
                output_cost = (completion_tokens + reasoning_tokens) * output_cost_per_token
                call_cost = input_cost + output_cost

                self.total_cost += call_cost

                if verbose:
                    print("\n" + "─" * 60)
                    print("📊 TOKEN USAGE")
                    print("─" * 60)
                    print(f"Prompt tokens:     {prompt_tokens:,}")
                    print(f"Completion tokens: {completion_tokens:,}")
                    if reasoning_tokens > 0:
                        print(f"Reasoning tokens:  {reasoning_tokens:,}")
                        print(f"├─ Reasoning:      {reasoning_tokens:,}")
                        print(f"└─ Output:         {completion_tokens:,}")
                    print(f"Total tokens:      {total_tokens:,}")

                    print(f"\n💰 COST")
                    print(f"Input cost:        ${input_cost:.6f}")
                    print(f"Output cost:       ${output_cost:.6f}")
                    print(f"Call cost:         ${call_cost:.6f}")

            if verbose:
                # Show final response
                print(f"\n✨ RESPONSE")
                print("─" * 60)
                print(f"Content: {content}")
                print("─" * 60 + "\n")

            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction

        except Exception as e:
            print(f"Error calling local LLM API: {e}")
            # Return random guess on error
            import random
            return random.randint(0, 1)

    async def _predict_single_async(self, jet: Any, verbose: bool = False) -> int:
        """
        Predict label for a single jet asynchronously.

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
        # Format jet and fill template
        prompt = fill_template(self.template, jet, self.format_type)

        # Call local LLM API asynchronously
        try:
            if verbose:
                print(f"\n🔧 API PARAMETERS")
                print(f"Model: {self.model_name}")
                print(f"Reasoning effort: {self.reasoning_effort}")
                print(f"Reasoning summary: {self.reasoning_summary}")
                print()

            response = await self.async_client.responses.create(
                model=self.model_name,
                instructions="You are a helpful assistant.",
                input=prompt,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary
                }
            )

            # Extract response content
            # The response.output is a list of ResponseReasoningItem objects
            # For reasoning models, the final answer is at the end of the reasoning text
            content = ""
            if hasattr(response, 'output') and isinstance(response.output, list):
                # Collect all reasoning text - the answer is typically at the end
                all_text = []
                for item in response.output:
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                all_text.append(content_item.text)

                # Combine all text - the final answer should be in there
                content = " ".join(all_text)

            # Convert to string and strip
            content = str(content).strip()

            if len(content) == 0:
                print("Warning: API returned empty content")
                print(f"Response object: {response}")
                return 0

            # Track token usage if available
            if hasattr(response, 'usage'):
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                completion_tokens = getattr(usage, 'completion_tokens', 0) or 0
                reasoning_tokens = getattr(usage, 'reasoning_tokens', 0) or 0
                total_tokens = getattr(usage, 'total_tokens', 0) or (prompt_tokens + completion_tokens + reasoning_tokens)

                # Update cumulative totals (thread-safe increment would be needed for true parallelism)
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_reasoning_tokens += reasoning_tokens

                # Calculate cost (example pricing - adjust based on actual costs)
                # Using similar rates to Gemini for consistency
                input_cost_per_token = 0.000000075
                output_cost_per_token = 0.0000003

                input_cost = prompt_tokens * input_cost_per_token
                output_cost = (completion_tokens + reasoning_tokens) * output_cost_per_token
                call_cost = input_cost + output_cost

                self.total_cost += call_cost

                if verbose:
                    print("\n" + "─" * 60)
                    print("📊 TOKEN USAGE")
                    print("─" * 60)
                    print(f"Prompt tokens:     {prompt_tokens:,}")
                    print(f"Completion tokens: {completion_tokens:,}")
                    if reasoning_tokens > 0:
                        print(f"Reasoning tokens:  {reasoning_tokens:,}")
                        print(f"├─ Reasoning:      {reasoning_tokens:,}")
                        print(f"└─ Output:         {completion_tokens:,}")
                    print(f"Total tokens:      {total_tokens:,}")

                    print(f"\n💰 COST")
                    print(f"Input cost:        ${input_cost:.6f}")
                    print(f"Output cost:       ${output_cost:.6f}")
                    print(f"Call cost:         ${call_cost:.6f}")

            if verbose:
                # Show final response
                print(f"\n✨ RESPONSE")
                print("─" * 60)
                print(f"Content: {content}")
                print("─" * 60 + "\n")

            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction

        except Exception as e:
            print(f"Error calling local LLM API (async): {e}")
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
        print(f"Warning: Could not parse prediction from response: '{response}'")
        return 0

    def _print_cumulative_stats(self) -> None:
        """Print cumulative token usage and cost statistics."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens + self.total_reasoning_tokens

        print("\n" + "═" * 60)
        print("📈 CUMULATIVE STATISTICS")
        print("═" * 60)
        print(f"Total prompt tokens:     {self.total_prompt_tokens:,}")
        print(f"Total completion tokens: {self.total_completion_tokens:,}")
        if self.total_reasoning_tokens > 0:
            print(f"Total reasoning tokens:  {self.total_reasoning_tokens:,}")
        print(f"Total tokens:            {total_tokens:,}")
        print(f"\n💰 Total estimated cost: ${self.total_cost:.6f}")
        print("═" * 60 + "\n")
