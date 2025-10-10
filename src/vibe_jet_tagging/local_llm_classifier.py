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

        # Track cumulative token usage and generation time
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0  # Reasoning/thinking tokens (estimated)
        self.total_generation_time = 0.0  # Total time spent in generation (seconds)

        # Persistent event loop for async operations
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Get or create a persistent event loop for async operations.

        Returns
        -------
        asyncio.AbstractEventLoop
            The persistent event loop
        """
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def _run_with_persistent_loop(
        self, X: list[Any], verbose: bool, max_concurrent: int
    ) -> list[int]:
        """
        Run async predictions using a persistent event loop.

        This avoids creating and destroying event loops for each predict() call,
        which prevents "Event loop is closed" errors when running sequential
        configurations.

        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed information
        max_concurrent : int
            Maximum number of concurrent requests

        Returns
        -------
        list[int]
            Predicted labels
        """
        loop = self._get_or_create_event_loop()
        return loop.run_until_complete(
            self._predict_async(X, verbose=verbose, max_concurrent=max_concurrent)
        )

    def close(self) -> None:
        """
        Clean up resources, including the persistent event loop and async client.

        Call this when you're completely done with the classifier to free resources.
        For sequential runs, you don't need to call this between predict() calls.
        """
        if self._event_loop is not None and not self._event_loop.is_closed():
            # Clean up async client first
            if self.async_client:
                try:
                    self._event_loop.run_until_complete(self.async_client.aclose())
                except Exception:
                    pass

            # Cancel any pending tasks
            try:
                pending = asyncio.all_tasks(self._event_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._event_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass

            # Close the loop
            try:
                self._event_loop.close()
            except Exception:
                pass

            self._event_loop = None

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

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

    def predict(
        self,
        X: list[Any],
        verbose: bool = False,
        use_async: bool = True,
        max_concurrent: int = 50
    ) -> list[int]:
        """
        Predict jet labels using local LLM.

        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and generation time information
        use_async : bool
            If True, use async/concurrent requests (default: True)
            Set to False for sequential processing
        max_concurrent : int
            Maximum number of concurrent requests (default: 50)
            Helps prevent overwhelming the server with large batches

        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predict")

        if use_async:
            # Use async for concurrent requests
            # Check if we're in a notebook with an existing event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in a notebook/existing event loop
                import nest_asyncio
                nest_asyncio.apply()
                predictions = asyncio.run(self._predict_async(X, verbose=verbose))
            except RuntimeError:
                # No running loop - use our persistent event loop
                predictions = self._run_with_persistent_loop(X, verbose, max_concurrent)
        else:
            # Sequential processing
            predictions = []
            for jet in X:
                prediction = self._predict_single(jet, verbose=verbose)
                predictions.append(prediction)

        if verbose:
            self._print_cumulative_stats()

        return predictions

    async def _predict_async(
        self, X: list[Any], verbose: bool = False, max_concurrent: int = 50
    ) -> list[int]:
        """
        Predict labels for multiple jets concurrently using async with batching.

        Parameters
        ----------
        X : list
            List of jets to classify
        verbose : bool
            If True, print detailed token usage and generation time information
        max_concurrent : int
            Maximum number of concurrent requests

        Returns
        -------
        list[int]
            Predicted labels (0 for gluon, 1 for quark)
        """
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_predict(jet):
            async with semaphore:
                return await self._predict_single_async(jet, verbose=verbose)

        # Create tasks with concurrency control
        tasks = [bounded_predict(jet) for jet in X]

        # Run all tasks with controlled concurrency
        predictions = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        results = []
        for i, pred in enumerate(predictions):
            if isinstance(pred, Exception):
                print(f"Warning: Error predicting jet {i}: {pred}")
                # Return random guess on error
                import random
                results.append(random.randint(0, 1))
            else:
                results.append(pred)

        return results

    def _predict_single(self, jet: Any, verbose: bool = False) -> int:
        """
        Predict label for a single jet.

        Parameters
        ----------
        jet : array-like
            Single jet data
        verbose : bool
            If True, print detailed token usage and generation time information

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

            import time
            start_time = time.time()
            response = self.client.responses.create(
                model=self.model_name,
                instructions="You are a helpful assistant.",
                input=prompt,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary
                }
            )
            generation_time = time.time() - start_time
            self.total_generation_time += generation_time

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

            # Extract reasoning trace (needed for token estimation, not just verbose output)
            reasoning_trace = None
            if hasattr(response, 'output') and isinstance(response.output, list):
                if len(response.output) > 0 and hasattr(response.output[0], 'content'):
                    if isinstance(response.output[0].content, list) and len(response.output[0].content) > 0:
                        if hasattr(response.output[0].content[0], 'text'):
                            reasoning_trace = response.output[0].content[0].text

            # Track token usage if available
            if hasattr(response, 'usage'):
                usage = response.usage

                # OpenAI Responses API uses input_tokens, output_tokens, total_tokens
                # (not prompt_tokens, completion_tokens, reasoning_tokens)
                input_tokens = getattr(usage, 'input_tokens', None)
                output_tokens = getattr(usage, 'output_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

                # Fallback to alternative names if needed
                if input_tokens is None:
                    input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                if output_tokens is None:
                    output_tokens = getattr(usage, 'completion_tokens', 0) or 0
                if total_tokens is None:
                    total_tokens = input_tokens + output_tokens

                # TODO: Improve reasoning token calculation. Currently estimated from character
                # lengths, but OpenAI Responses API should provide accurate reasoning token counts.
                # Check if the API returns reasoning_tokens separately in usage object.

                # Try to get reasoning tokens directly from API
                reasoning_tokens = getattr(usage, 'reasoning_tokens', None)

                if reasoning_tokens is None:
                    # Fallback: estimate from character lengths if reasoning trace exists
                    reasoning_tokens = 0
                    completion_tokens_est = output_tokens

                    if reasoning_trace and content:
                        reasoning_chars = len(reasoning_trace)
                        completion_chars = len(content)
                        total_chars = reasoning_chars + completion_chars

                        if total_chars > 0:
                            # Split output_tokens proportionally based on character length
                            reasoning_ratio = reasoning_chars / total_chars
                            reasoning_tokens = int(output_tokens * reasoning_ratio)
                            completion_tokens_est = output_tokens - reasoning_tokens
                else:
                    # API provided reasoning tokens directly
                    completion_tokens_est = output_tokens - reasoning_tokens

                # Update cumulative totals
                self.total_prompt_tokens += input_tokens
                self.total_completion_tokens += completion_tokens_est
                self.total_reasoning_tokens += reasoning_tokens

                if verbose:
                    print("\n" + "─" * 60)
                    print("📊 TOKEN USAGE")
                    print("─" * 60)
                    print(f"Input tokens:        {input_tokens:,}")
                    print(f"Output tokens:       {output_tokens:,}")
                    if reasoning_tokens > 0:
                        print(f"  ├─ Reasoning (est): {reasoning_tokens:,}")
                        print(f"  └─ Completion:      {completion_tokens_est:,}")
                    print(f"Total tokens:        {total_tokens:,}")

                    print(f"\n⏱️  GENERATION TIME")
                    print(f"Generation time:     {generation_time:.3f}s")

            if verbose:
                # Show reasoning trace if available
                if reasoning_trace:
                    print(f"\n🧠 REASONING TRACE")
                    print("─" * 60)
                    # Show preview (first 300 chars) or full if short
                    if len(reasoning_trace) > 300:
                        print(f"{reasoning_trace[:300]}...")
                        print(f"[... {len(reasoning_trace) - 300} more characters ...]")
                    else:
                        print(reasoning_trace)
                    print("─" * 60)

                # Show final response
                print(f"\n✨ FINAL OUTPUT")
                print("─" * 60)
                print(f"{content}")
                print("─" * 60 + "\n")

            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction

        except Exception as e:
            print(f"Error calling local LLM API: {e}")
            # Return random guess on error
            import random
            return random.randint(0, 1)

    async def _predict_single_async(
        self, jet: Any, verbose: bool = False, max_retries: int = 3
    ) -> int:
        """
        Predict label for a single jet asynchronously with retry logic.

        Parameters
        ----------
        jet : array-like
            Single jet data
        verbose : bool
            If True, print detailed token usage and generation time information
        max_retries : int
            Maximum number of retry attempts on connection errors

        Returns
        -------
        int
            Predicted label (0 or 1)
        """
        # Format jet and fill template
        prompt = fill_template(self.template, jet, self.format_type)

        # Retry logic for connection errors
        for attempt in range(max_retries):
            try:
                return await self._predict_single_async_impl(jet, prompt, verbose)
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's a connection error worth retrying
                if any(keyword in error_msg for keyword in ['connection', 'timeout', 'refused']):
                    if attempt < max_retries - 1:
                        # Wait before retrying (exponential backoff)
                        wait_time = 0.5 * (2 ** attempt)
                        if verbose:
                            print(f"Connection error, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"Error calling local LLM API (async): {e}")
                        import random
                        return random.randint(0, 1)
                else:
                    # Not a connection error, don't retry
                    print(f"Error calling local LLM API (async): {e}")
                    import random
                    return random.randint(0, 1)

        # Should not reach here, but just in case
        import random
        return random.randint(0, 1)

    async def _predict_single_async_impl(
        self, jet: Any, prompt: str, verbose: bool = False
    ) -> int:
        """
        Implementation of async prediction for a single jet.

        Parameters
        ----------
        jet : array-like
            Single jet data
        prompt : str
            Formatted prompt
        verbose : bool
            If True, print detailed information

        Returns
        -------
        int
            Predicted label (0 or 1)
        """
        try:
            if verbose:
                print(f"\n🔧 API PARAMETERS")
                print(f"Model: {self.model_name}")
                print(f"Reasoning effort: {self.reasoning_effort}")
                print(f"Reasoning summary: {self.reasoning_summary}")
                print()

            import time
            start_time = time.time()
            response = await self.async_client.responses.create(
                model=self.model_name,
                instructions="You are a helpful assistant.",
                input=prompt,
                reasoning={
                    "effort": self.reasoning_effort,
                    "summary": self.reasoning_summary
                }
            )
            generation_time = time.time() - start_time
            self.total_generation_time += generation_time

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

            # Extract reasoning trace (needed for token estimation, not just verbose output)
            reasoning_trace = None
            if hasattr(response, 'output') and isinstance(response.output, list):
                if len(response.output) > 0 and hasattr(response.output[0], 'content'):
                    if isinstance(response.output[0].content, list) and len(response.output[0].content) > 0:
                        if hasattr(response.output[0].content[0], 'text'):
                            reasoning_trace = response.output[0].content[0].text

            # Track token usage if available
            if hasattr(response, 'usage'):
                usage = response.usage

                # OpenAI Responses API uses input_tokens, output_tokens, total_tokens
                input_tokens = getattr(usage, 'input_tokens', None)
                output_tokens = getattr(usage, 'output_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

                # Fallback to alternative names if needed
                if input_tokens is None:
                    input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
                if output_tokens is None:
                    output_tokens = getattr(usage, 'completion_tokens', 0) or 0
                if total_tokens is None:
                    total_tokens = input_tokens + output_tokens

                # TODO: Improve reasoning token calculation. Currently estimated from character
                # lengths, but OpenAI Responses API should provide accurate reasoning token counts.
                # Check if the API returns reasoning_tokens separately in usage object.

                # Try to get reasoning tokens directly from API
                reasoning_tokens = getattr(usage, 'reasoning_tokens', None)

                if reasoning_tokens is None:
                    # Fallback: estimate from character lengths if reasoning trace exists
                    reasoning_tokens = 0
                    completion_tokens_est = output_tokens

                    if reasoning_trace and content:
                        reasoning_chars = len(reasoning_trace)
                        completion_chars = len(content)
                        total_chars = reasoning_chars + completion_chars

                        if total_chars > 0:
                            # Split output_tokens proportionally based on character length
                            reasoning_ratio = reasoning_chars / total_chars
                            reasoning_tokens = int(output_tokens * reasoning_ratio)
                            completion_tokens_est = output_tokens - reasoning_tokens
                else:
                    # API provided reasoning tokens directly
                    completion_tokens_est = output_tokens - reasoning_tokens

                # Update cumulative totals (thread-safe increment would be needed for true parallelism)
                self.total_prompt_tokens += input_tokens
                self.total_completion_tokens += completion_tokens_est
                self.total_reasoning_tokens += reasoning_tokens

                if verbose:
                    print("\n" + "─" * 60)
                    print("📊 TOKEN USAGE")
                    print("─" * 60)
                    print(f"Input tokens:        {input_tokens:,}")
                    print(f"Output tokens:       {output_tokens:,}")
                    if reasoning_tokens > 0:
                        print(f"  ├─ Reasoning (est): {reasoning_tokens:,}")
                        print(f"  └─ Completion:      {completion_tokens_est:,}")
                    print(f"Total tokens:        {total_tokens:,}")

                    print(f"\n⏱️  GENERATION TIME")
                    print(f"Generation time:     {generation_time:.3f}s")

            if verbose:
                # Show reasoning trace if available
                if reasoning_trace:
                    print(f"\n🧠 REASONING TRACE")
                    print("─" * 60)
                    if len(reasoning_trace) > 300:
                        print(f"{reasoning_trace[:300]}...")
                        print(f"[... {len(reasoning_trace) - 300} more characters ...]")
                    else:
                        print(reasoning_trace)
                    print("─" * 60)

                # Show final response
                print(f"\n✨ FINAL OUTPUT")
                print("─" * 60)
                print(f"{content}")
                print("─" * 60 + "\n")

            # Parse prediction
            prediction = self._parse_prediction(content)
            return prediction

        except Exception as e:
            # Re-raise the exception to be handled by retry logic
            raise

    def _parse_prediction(self, response: str) -> int:
        """
        Parse LLM response to extract 0 or 1.

        The model typically provides reasoning followed by a final answer.
        We look for the LAST occurrence of 0 or 1, which is usually the final answer.

        Parameters
        ----------
        response : str
            LLM response text

        Returns
        -------
        int
            Parsed label (0 or 1)
        """
        # Strategy 1: Look for common answer patterns at the end
        # Patterns like "answer: 1", "label: 0", "prediction: 1", or just "1" at the end
        end_patterns = [
            r'(?:answer|label|prediction|output|result)[:=\s]+([01])\b',
            r'\b([01])\s*$',  # 0 or 1 at the very end
            r'(?:thus|therefore|so|hence).*?([01])\b',  # After conclusion words
        ]

        for pattern in end_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                # Get the LAST match
                return int(matches[-1].group(1))

        # Strategy 2: Find ALL occurrences of 0 or 1, take the last one
        all_matches = list(re.finditer(r'\b[01]\b', response))
        if all_matches:
            # Return the LAST occurrence (most likely the final answer)
            return int(all_matches[-1].group())

        # Strategy 3: Check for keywords (fallback)
        response_lower = response.lower()
        if "quark" in response_lower:
            return 1
        elif "gluon" in response_lower:
            return 0

        # Default to 0 if unclear
        print(f"Warning: Could not parse prediction from response: '{response[:100]}...'")
        return 0

    def _print_cumulative_stats(self) -> None:
        """Print cumulative token usage and generation time statistics."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens + self.total_reasoning_tokens

        print("\n" + "═" * 60)
        print("📈 CUMULATIVE STATISTICS")
        print("═" * 60)
        print(f"Total prompt tokens:     {self.total_prompt_tokens:,}")
        print(f"Total completion tokens: {self.total_completion_tokens:,}")
        if self.total_reasoning_tokens > 0:
            print(f"Total reasoning tokens:  {self.total_reasoning_tokens:,} (estimate)")
        print(f"Total tokens:            {total_tokens:,}")
        print(f"\n⏱️  Total generation time: {self.total_generation_time:.2f}s")
        print("═" * 60 + "\n")
