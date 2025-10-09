# Local LLM Classifier - Quick Start Guide

Get started with `LocalLLMClassifier` in 5 minutes!

## Prerequisites

1. **Local LLM server running**:
   ```bash
   vllm serve openai/gpt-oss-120b --port 8000
   ```

2. **Data downloaded**:
   ```bash
   uv run python scripts/download_qg_data.py -n 10000
   ```

## Basic Usage

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
    reasoning_effort="medium",  # "low", "medium", or "high"
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Fit (no-op for zero-shot)
clf.fit([], [])

# Predict (async by default - 5x faster!)
predictions = clf.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, predictions)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
```

## Verbose Mode (Single Jet)

```python
# Get detailed output for one jet
pred = clf.predict([X_test[0]], verbose=True, use_async=False)

# Shows:
# - Token usage (input/output/reasoning)
# - Reasoning trace preview
# - Final output
# - Cost estimate
```

## Jupyter Notebook

Works seamlessly! Async support auto-detects notebook environment.

```python
# In a Jupyter cell
clf = LocalLLMClassifier(reasoning_effort="low")
clf.fit([], [])

# Async works automatically!
predictions = clf.predict(X_test, use_async=True)
```

## Reasoning Effort Levels

| Effort | Use When | Speed | Quality |
|--------|----------|-------|---------|
| `"low"` | Quick testing | Fast (~2.5s/jet) | Good |
| `"medium"` | **Recommended** | Medium (~3.5s/jet) | Better |
| `"high"` | Best accuracy | Slow (~12s/jet) | Best |

## Sequential vs Async

```python
# Sequential: Slow but simple
pred_seq = clf.predict(X_test, use_async=False)

# Async: 5x faster! (default)
pred_async = clf.predict(X_test, use_async=True)
```

## Common Tasks

### Compare Different Efforts

```python
for effort in ["low", "medium", "high"]:
    clf = LocalLLMClassifier(reasoning_effort=effort)
    clf.fit([], [])
    pred = clf.predict(X_test[:10])
    acc = accuracy_score(y_test[:10], pred)
    print(f"{effort}: {acc:.2%} accuracy")
```

### Get Token Statistics

```python
clf.predict(X_test, verbose=False)

print(f"Total input tokens: {clf.total_prompt_tokens}")
print(f"Total reasoning tokens: {clf.total_reasoning_tokens}")
print(f"Total completion tokens: {clf.total_completion_tokens}")
print(f"Total cost: ${clf.total_cost:.4f}")
```

### Preview Prompt

```python
clf.preview_prompt(X_test[0])
# Shows exactly what will be sent to the model
```

## Troubleshooting

### Server Not Running

```python
import requests
response = requests.get("http://localhost:8000/v1/models")
print(response.json())  # Should show available models
```

### Async Not Working in Notebook

The classifier auto-handles this, but if you see errors:
```python
import nest_asyncio
nest_asyncio.apply()
```

### Parser Not Finding Answer

Check verbose output:
```python
pred = clf.predict([X_test[0]], verbose=True)
# Look at "REASONING TRACE" and "FINAL OUTPUT"
```

## Performance Tips

1. **Use async mode**: 5x faster
2. **Start with "low" effort**: Test quickly
3. **Batch predictions**: Process multiple jets at once
4. **Monitor tokens**: Check `clf.total_*` after predictions

## Next Steps

- Read full documentation: `notes/local_llm_integration.md`
- See examples: `notebooks/2025-10-09-IT-gpt-oss-classifier.ipynb`
- Understand API: `notes/response_structure_guide.md`

## Example Output

```
Accuracy: 0.720
AUC: 0.782

Total tokens: 108,450
Total cost: $0.0234
Average time: 0.74s per jet (async)
```

**Happy tagging! 🎯**
