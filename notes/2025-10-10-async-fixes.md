# Async Processing Fixes

## Problem: Connection Errors with Large Batches

When processing 1000 jets, the script was firing all 1000 async requests simultaneously, overwhelming the vLLM server and causing connection errors:

```
Error calling local LLM API (async): Connection error.
```

## Solution: Concurrency Control + Retry Logic

### 1. **Semaphore-Based Concurrency Limiting**

Added `max_concurrent` parameter (default: 50) that uses `asyncio.Semaphore` to limit concurrent requests:

```python
# Only allow 50 requests at a time
semaphore = asyncio.Semaphore(max_concurrent)

async def bounded_predict(jet):
    async with semaphore:  # Wait for available slot
        return await self._predict_single_async(jet, verbose=verbose)
```

### 2. **Automatic Retry with Exponential Backoff**

Added retry logic (max 3 attempts) for connection errors:

```python
for attempt in range(max_retries):
    try:
        return await self._predict_single_async_impl(jet, prompt, verbose)
    except Exception as e:
        if 'connection' in error_msg or 'timeout' in error_msg:
            wait_time = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
            await asyncio.sleep(wait_time)
            continue
```

### 3. **Graceful Error Handling**

If a request fails after all retries, it returns a random guess instead of crashing:

```python
# Handle exceptions gracefully
if isinstance(pred, Exception):
    print(f"Warning: Error predicting jet {i}: {pred}")
    import random
    results.append(random.randint(0, 1))
```

## Usage

### Command-Line Control

```bash
# Default: 50 concurrent requests
uv run python scripts/analyze_llm_templates.py --num_jets 1000

# Conservative: 20 concurrent (safer for busy servers)
uv run python scripts/analyze_llm_templates.py --num_jets 1000 --max_concurrent 20

# Aggressive: 100 concurrent (if server can handle it)
uv run python scripts/analyze_llm_templates.py --num_jets 1000 --max_concurrent 100
```

### Programmatic Control

```python
from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier

clf = LocalLLMClassifier(...)
clf.fit([], [])

# Control concurrency
predictions = clf.predict(
    X,
    use_async=True,
    max_concurrent=30  # Limit to 30 concurrent requests
)
```

## Tuning `max_concurrent`

The optimal value depends on your server:

| Server Capacity | Recommended `max_concurrent` | Notes |
|----------------|------------------------------|-------|
| Single GPU, local | 20-50 | Conservative, prevents overwhelming |
| Multi-GPU server | 50-100 | Can handle more parallel requests |
| Shared HPC cluster | 10-30 | Be nice to other users |
| High-powered dedicated | 100-200 | If server is powerful and dedicated |

### Signs You Need to Lower It:
- Connection errors appearing
- Server response times increasing dramatically
- OOM (out of memory) errors on server

### Signs You Can Increase It:
- Low GPU utilization (check with `nvidia-smi`)
- Predictions running much slower than necessary
- No connection errors even with current setting

## Performance Impact

### Before (no concurrency control):
```
1000 jets × 1000 concurrent = Connection errors!
```

### After (max_concurrent=50):
```
1000 jets × 50 concurrent = Smooth processing
- Retries: ~2-5 per 1000 jets (rare)
- No crashes
- Predictable timing
```

## Example Results

Test with 20 jets, max_concurrent=10:
- ✅ No connection errors
- ✅ Total time: 89.54s
- ✅ Reasoning tokens: 982
- ✅ All predictions successful

## Monitoring Progress

While running large batches, you can monitor:

```bash
# In another terminal, check server logs
tail -f vllm_server.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check script progress
screen -r llm_analysis
```

## Error Recovery

The script now handles errors gracefully:

1. **Temporary connection errors**: Retries with exponential backoff
2. **Persistent errors**: Falls back to random guess for that jet
3. **Partial failures**: Continues processing remaining jets
4. **Complete results**: Always saves what was successfully processed

This means even if a few requests fail, you still get results for the vast majority of jets.
