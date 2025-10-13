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

        # Store connection parameters
        self.base_url = base_url
        self.api_key = api_key

        # Set up sync OpenAI client
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        # Create async client lazily (only when needed)
        self._async_client: Optional[AsyncOpenAI] = None

        self.is_fitted = False

        # Track cumulative token usage and generation time
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0  # Reasoning/thinking tokens (estimated)
        self.total_generation_time = 0.0  # Total time spent in generation (seconds)
        self.failed_predictions = 0  # Track failed predictions

        # Persistent event loop for async operations
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def async_client(self) -> AsyncOpenAI:
        """
        Get or create the async OpenAI client.

        The client is created lazily to avoid cleanup issues when
        multiple classifier instances are created sequentially.

        Returns
        -------
        AsyncOpenAI
            The async client
        """
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._async_client

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
    ) -> list[float]:
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
        list[float]
            Predicted probabilities
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
        # Clean up async client first (if it was created)
        if self._async_client is not None:
            try:
                # Close the async client synchronously
                import asyncio
                # Check if there's a running loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, schedule the close
                    asyncio.create_task(self._async_client.aclose())
                except RuntimeError:
                    # No running loop, use our persistent loop if available
                    if self._event_loop is not None and not self._event_loop.is_closed():
                        self._event_loop.run_until_complete(self._async_client.aclose())
                    else:
                        # Create a temporary loop just for cleanup
                        temp_loop = asyncio.new_event_loop()
                        try:
                            temp_loop.run_until_complete(self._async_client.aclose())
                        finally:
                            temp_loop.close()
            except Exception:
                pass
            finally:
                self._async_client = None

        # Clean up event loop
        if self._event_loop is not None and not self._event_loop.is_closed():
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
        """
        Cleanup on deletion.

        Note: We don't call close() here because it can cause issues when
        multiple classifier instances are garbage collected while event loops
        are active. The async client will be cleaned up naturally by Python's
        garbage collector.
        """
        # Don't call close() - let Python handle cleanup
        # This prevents "Event loop is closed" errors during garbage collection
        pass

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
    ) -> list[float]:
        """
        Predict jet probabilities using local LLM.

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
        list[float]
            Predicted probabilities (0.0 to 1.0, where 1.0 = quark, 0.0 = gluon)
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
    ) -> list[float]:
        """
        Predict probabilities for multiple jets concurrently using async with batching.

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
        list[float]
            Predicted probabilities (0.0 to 1.0, where 1.0 = quark, 0.0 = gluon)
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
        failed_count = 0
        for i, pred in enumerate(predictions):
            if isinstance(pred, Exception):
                print(f"Warning: Error predicting jet {i}: {pred}")
                # Return uncertain probability on error
                results.append(0.5)
                failed_count += 1
            else:
                results.append(pred)

        # Update total failed predictions count
        self.failed_predictions += failed_count

        return results

    def _predict_single(self, jet: Any, verbose: bool = False) -> float:
        """
        Predict probability for a single jet.

        Parameters
        ----------
        jet : array-like
            Single jet data
        verbose : bool
            If True, print detailed token usage and generation time information

        Returns
        -------
        float
            Predicted probability (0.0 to 1.0, where 1.0 = quark, 0.0 = gluon)
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

                # Get reasoning tokens from the correct location in the API response.
                # The OpenAI Responses API returns reasoning tokens in:
                #   response.usage.output_tokens_details.reasoning_tokens
                # NOT in response.usage.reasoning_tokens (which doesn't exist)
                
                reasoning_tokens = 0
                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0) or 0
                
                # Completion tokens = total output tokens - reasoning tokens
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
                        print(f"  ├─ Reasoning:       {reasoning_tokens:,}")
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
            # Return uncertain probability on error
            self.failed_predictions += 1
            return 0.5

    async def _predict_single_async(
        self, jet: Any, verbose: bool = False, max_retries: int = 3
    ) -> float:
        """
        Predict probability for a single jet asynchronously with retry logic.

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
        float
            Predicted probability (0.0 to 1.0, where 1.0 = quark, 0.0 = gluon)
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
                        self.failed_predictions += 1
                        return 0.5
                else:
                    # Not a connection error, don't retry
                    print(f"Error calling local LLM API (async): {e}")
                    self.failed_predictions += 1
                    return 0.5

        # Should not reach here, but just in case
        self.failed_predictions += 1
        return 0.5

    async def _predict_single_async_impl(
        self, jet: Any, prompt: str, verbose: bool = False
    ) -> float:
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
        float
            Predicted probability (0.0 to 1.0, where 1.0 = quark, 0.0 = gluon)
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

                # Get reasoning tokens from the correct location in the API response.
                # The OpenAI Responses API returns reasoning tokens in:
                #   response.usage.output_tokens_details.reasoning_tokens
                # NOT in response.usage.reasoning_tokens (which doesn't exist)
                
                reasoning_tokens = 0
                if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                    reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0) or 0
                
                # Completion tokens = total output tokens - reasoning tokens
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
                        print(f"  ├─ Reasoning:       {reasoning_tokens:,}")
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

    def _parse_prediction(self, response: str) -> float:
        """
        Parse LLM response to extract probability value between 0 and 1.

        The model should provide a probability where:
        - 0.0 means definitely a gluon jet
        - 1.0 means definitely a quark jet
        - Values in between represent confidence

        Parameters
        ----------
        response : str
            LLM response text

        Returns
        -------
        float
            Parsed probability (0.0 to 1.0)
        """
        # Strategy 1: Look for decimal numbers between 0 and 1
        # Match patterns like "0.75", "0.8", ".95", etc.
        decimal_patterns = [
            r'(?:probability|confidence|answer|result)[:=\s]+([0-9]*\.?[0-9]+)',
            r'\b(0\.[0-9]+|1\.0+|0\.0+)\b',  # Decimal between 0 and 1
            r'\b(0|1)\.[0-9]+\b',  # 0.x or 1.x
            r'(?:^|\s)([0-9]*\.?[0-9]+)(?:\s|$)',  # Any number near word boundaries
        ]

        for pattern in decimal_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                # Get the LAST match (most likely the final answer)
                try:
                    value = float(matches[-1].group(1))
                    # Check if it's in valid range
                    if 0.0 <= value <= 1.0:
                        return value
                except (ValueError, IndexError):
                    continue

        # Strategy 2: Look for standalone decimals at the end of response
        # This catches cases where the model just outputs "0.75" or similar
        end_match = re.search(r'([0-9]*\.?[0-9]+)\s*$', response.strip())
        if end_match:
            try:
                value = float(end_match.group(1))
                if 0.0 <= value <= 1.0:
                    return value
            except ValueError:
                pass

        # Strategy 3: Check for binary 0 or 1 (fallback for models that still output binary)
        binary_match = re.search(r'\b([01])\b(?!\.)', response)
        if binary_match:
            return float(binary_match.group(1))

        # Strategy 4: Check for keywords (last resort fallback)
        response_lower = response.lower()
        if "definitely" in response_lower or "certainly" in response_lower:
            if "quark" in response_lower:
                return 1.0
            elif "gluon" in response_lower:
                return 0.0

        if "quark" in response_lower and "gluon" not in response_lower:
            return 0.8  # Favor quark but not certain
        elif "gluon" in response_lower and "quark" not in response_lower:
            return 0.2  # Favor gluon but not certain

        # Default to 0.5 (uncertain) if we can't parse
        print(f"Warning: Could not parse probability from response: '{response[:100]}...'")
        return 0.5

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
        
        # Show failed predictions warning if any
        if self.failed_predictions > 0:
            print(f"\n⚠️  FAILED PREDICTIONS: {self.failed_predictions:,}")
            print(f"   (replaced with random guesses)")
        
        print("═" * 60 + "\n")
