# Local LLM Integration

This document describes the integration of local LLM support via OpenAI-compatible APIs.

## Overview

We've added `LocalLLMClassifier` that works with local OpenAI-compatible inference servers (like vLLM, text-generation-inference, etc.) running reasoning-capable models.

## Key Features

### 1. Reasoning Effort Control

Unlike Gemini's token-based `thinking_budget`, local reasoning models use effort levels:

```python
classifier = LocalLLMClassifier(
    reasoning_effort="medium",  # Options: "low", "medium", "high"
    reasoning_summary="auto"     # Summary detail level
)
```

**Effort Levels:**
- `"low"`: Fast reasoning with minimal depth (~1,250 tokens)
- `"medium"`: Balanced reasoning (~1,760 tokens)
- `"high"`: Deep reasoning with maximum effort (~7,540 tokens)

### 2. Async/Concurrent Processing

The classifier supports both sequential and concurrent request processing:

```python
# Async mode (default, 5.1x faster)
predictions = classifier.predict(X_test, use_async=True)

# Sequential mode
predictions = classifier.predict(X_test, use_async=False)
```

**Performance Comparison (5 jets, medium effort):**
- Async: 3.70 seconds (0.74 sec/jet)
- Sequential: 18.97 seconds (3.79 sec/jet)
- **Speedup: 5.1x**

### 3. OpenAI-Compatible API

Works with any OpenAI-compatible inference server:

```python
from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier

classifier = LocalLLMClassifier(
    model_name="openai/gpt-oss-120b",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # Or actual key if required
)
```

## Architecture

### Client Setup

The classifier maintains both sync and async clients:

```python
self.client = OpenAI(base_url=base_url, api_key=api_key)
self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
```

### Response Processing

The local API returns `ResponseReasoningItem` objects:

```python
response.output = [
    ResponseReasoningItem(
        type='reasoning',
        content=[
            Content(
                text="reasoning text with final answer...",
                type='reasoning_text'
            )
        ]
    )
]
```

The classifier extracts the reasoning text and parses the final answer (0 or 1).

### Async Implementation

Uses `asyncio.gather()` to run multiple predictions concurrently:

```python
async def _predict_async(self, X: list[Any], verbose: bool = False) -> list[int]:
    tasks = [self._predict_single_async(jet, verbose) for jet in X]
    predictions = await asyncio.gather(*tasks)
    return list(predictions)
```

## Usage Example

```python
from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier
import numpy as np

# Load data
data = np.load('data/qg_jets.npz')
X_test = data['X'][:100]
y_test = data['y'][:100]

# Create classifier
clf = LocalLLMClassifier(
    model_name="openai/gpt-oss-120b",
    template_name="simple_list",
    format_type="list",
    reasoning_effort="medium",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Fit (no-op for zero-shot)
clf.fit([], [])

# Predict with async (default)
predictions = clf.predict(X_test, verbose=False, use_async=True)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
```

## Testing

Run the test script:

```bash
# Make sure your local LLM server is running
# E.g., vLLM with: vllm serve openai/gpt-oss-120b --port 8000

uv run python scripts/test_local_llm.py
```

## Comparison with LLMClassifier (Gemini)

| Feature | LocalLLMClassifier | LLMClassifier (Gemini) |
|---------|-------------------|------------------------|
| API | OpenAI-compatible | Google Gemini |
| Reasoning Control | effort ("low"/"medium"/"high") | thinking_budget (token count) |
| Async Support | ✅ Built-in | ❌ Not implemented |
| Cost | Local (free) | Paid API |
| Speed | Very fast with async | Depends on API |
| Models | Any compatible model | Gemini 2.5 series |

## Notes

- Token usage tracking may vary depending on the server implementation
- Cost calculations are placeholder values (adjust based on actual costs)
- Cumulative token tracking is not thread-safe; consider using locks for true parallelism
- The async client automatically handles connection pooling and request batching

## Future Improvements

1. Add thread-safe token counting with locks
2. Support batch API endpoints if available
3. Add retry logic with exponential backoff
4. Implement caching for repeated jets
5. Add support for few-shot examples via the `instructions` parameter
