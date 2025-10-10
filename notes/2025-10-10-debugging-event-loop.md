# Debugging Plan: Event Loop Closure Issue

**Date**: 2025-10-10
**Status**: ACTIVE - Event loop still closing despite fixes

## Problem Summary

Getting `RuntimeError: Event loop is closed` when running sequential configurations in `run_llm_analysis.py`, even after:
1. Implementing persistent event loop
2. Implementing lazy async client creation
3. Removing cleanup from `__del__`

**Stack trace shows**: httpx's AsyncClient trying to call `loop.call_soon()` on a closed loop during its cleanup.

## Debugging Steps

### Phase 1: Isolate the Issue (30 min)

#### 1.1 Minimal Reproduction Test
Create `scripts/debug_event_loop.py`:

```python
"""Minimal test to reproduce event loop closure issue."""
import numpy as np
from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier

# Load minimal data
data = np.load("data/qg_jets.npz")
X = data["X"][:10]  # Just 10 jets
y = data["y"][:10]

configs = [
    ("simple_list", "low"),
    ("simple_list", "low"),  # Same config twice
]

print("Testing sequential classifier runs...")

for i, (template, effort) in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Config: {template} | {effort}")

    clf = LocalLLMClassifier(
        model_name="openai/gpt-oss-120b",
        template_name=template,
        reasoning_effort=effort,
    )
    clf.fit(X, y)

    # Add detailed logging
    print(f"  Before predict:")
    print(f"    - Event loop: {clf._event_loop}")
    print(f"    - Async client: {clf._async_client}")

    try:
        predictions = clf.predict(X, verbose=False, use_async=True, max_concurrent=10)

        print(f"  After predict:")
        print(f"    - Event loop: {clf._event_loop}")
        print(f"    - Event loop closed: {clf._event_loop.is_closed() if clf._event_loop else 'N/A'}")
        print(f"    - Async client: {clf._async_client}")
        print(f"    - Predictions: {len(predictions)} made")

    except Exception as e:
        print(f"  ERROR during predict: {e}")
        import traceback
        traceback.print_exc()
        break

    print(f"  Deleting classifier...")
    del clf

    import gc
    gc.collect()
    print(f"  GC complete")

print("\nTest complete")
```

**Run**: `uv run python scripts/debug_event_loop.py`

**Questions to answer**:
- Does the error happen on the 1st or 2nd config?
- What is the state of the event loop before/after each predict()?
- Does explicit `del clf` + `gc.collect()` trigger the issue?

#### 1.2 Test Without Deletion
Modify test to keep all classifiers in memory:

```python
classifiers = []
for i, (template, effort) in enumerate(configs, 1):
    clf = LocalLLMClassifier(...)
    predictions = clf.predict(...)
    classifiers.append(clf)  # Keep reference
    # No deletion until the very end
```

**Question**: Does the error still occur if we don't garbage collect between runs?

### Phase 2: Inspect Event Loop Lifecycle (20 min)

#### 2.1 Add Instrumentation
Temporarily modify `local_llm_classifier.py` to log all event loop operations:

```python
def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
    import sys
    if self._event_loop is None or self._event_loop.is_closed():
        print(f"[DEBUG] Creating NEW event loop (id={id(self)})", file=sys.stderr)
        self._event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._event_loop)
    else:
        print(f"[DEBUG] Reusing EXISTING event loop (id={id(self)})", file=sys.stderr)
    print(f"[DEBUG] Loop state: closed={self._event_loop.is_closed()}", file=sys.stderr)
    return self._event_loop
```

Also instrument `_run_with_persistent_loop`:

```python
def _run_with_persistent_loop(self, X, verbose, max_concurrent):
    import sys
    print(f"[DEBUG] _run_with_persistent_loop START (clf id={id(self)})", file=sys.stderr)
    loop = self._get_or_create_event_loop()
    print(f"[DEBUG] Got loop: {loop}, closed={loop.is_closed()}", file=sys.stderr)

    result = loop.run_until_complete(
        self._predict_async(X, verbose=verbose, max_concurrent=max_concurrent)
    )

    print(f"[DEBUG] _run_with_persistent_loop END, loop closed={loop.is_closed()}", file=sys.stderr)
    return result
```

**Run**: `uv run python scripts/debug_event_loop.py 2>&1 | tee debug.log`

**Look for**:
- Is a new loop created for each classifier?
- Is the loop closed between predict() calls?
- What's the timing of loop closure vs AsyncClient cleanup?

### Phase 3: Inspect AsyncClient Lifecycle (20 min)

#### 3.1 Add AsyncClient Instrumentation
Add to the `async_client` property:

```python
@property
def async_client(self) -> AsyncOpenAI:
    import sys
    if self._async_client is None:
        print(f"[DEBUG] Creating NEW async client (clf id={id(self)})", file=sys.stderr)
        self._async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        print(f"[DEBUG] Async client created: {self._async_client}", file=sys.stderr)
    return self._async_client
```

Add to `__del__`:

```python
def __del__(self):
    import sys
    print(f"[DEBUG] __del__ called for classifier id={id(self)}", file=sys.stderr)
    print(f"[DEBUG]   - _async_client: {self._async_client}", file=sys.stderr)
    print(f"[DEBUG]   - _event_loop: {self._event_loop}", file=sys.stderr)
    if self._event_loop:
        print(f"[DEBUG]   - loop closed: {self._event_loop.is_closed()}", file=sys.stderr)
```

**Run**: `uv run python scripts/debug_event_loop.py 2>&1 | tee debug.log`

**Look for**:
- When is the async client created?
- When does `__del__` get called?
- What's the state of the loop when `__del__` is called?

### Phase 4: Test Alternative Approaches (30 min)

#### 4.1 Option A: Never Close the Loop
Test if the issue is loop closure itself:

```python
# In _run_with_persistent_loop, never close the loop
# See if errors still occur
```

#### 4.2 Option B: One Global Event Loop
Test using a single global event loop for all classifiers:

```python
# At module level in local_llm_classifier.py
_GLOBAL_EVENT_LOOP = None

def get_global_event_loop():
    global _GLOBAL_EVENT_LOOP
    if _GLOBAL_EVENT_LOOP is None or _GLOBAL_EVENT_LOOP.is_closed():
        _GLOBAL_EVENT_LOOP = asyncio.new_event_loop()
    return _GLOBAL_EVENT_LOOP

# Use this instead of instance-level loop
```

#### 4.3 Option C: Use asyncio.run() with Context Manager
Test if the issue is loop reuse:

```python
async def _predict_with_context(self, X, verbose, max_concurrent):
    async with AsyncOpenAI(...) as client:
        # Use context manager for proper cleanup
        return await self._predict_async(X, verbose, max_concurrent)

# Then in predict():
predictions = asyncio.run(self._predict_with_context(...))
```

#### 4.4 Option D: Force Sync Mode
Test if async is even necessary:

```python
# Temporarily force use_async=False
# See if sequential sync calls work fine
```

### Phase 5: Inspect Python/httpx Internals (20 min)

#### 5.1 Check Python Version Compatibility
```bash
python --version  # Check if 3.13.2 has known asyncio issues
```

Search for:
- "Python 3.13 asyncio event loop closed"
- "httpx asyncio loop closed error"
- "openai-python asyncio cleanup"

#### 5.2 Check httpx Version
```bash
uv run python -c "import httpx; print(httpx.__version__)"
```

Test with different httpx version if needed:
```bash
uv add "httpx==0.25.0"  # Try an older stable version
```

#### 5.3 Check openai-python Version
```bash
uv run python -c "import openai; print(openai.__version__)"
```

### Phase 6: Environment-Specific Issues (15 min)

#### 6.1 Test Concurrency Level
Does the error depend on `max_concurrent`?

```python
# Test with max_concurrent = 1, 10, 50, 100, 300
# See if higher concurrency triggers it faster
```

#### 6.2 Test Jet Count
Does the error depend on number of jets?

```python
# Test with 5, 10, 50, 100 jets
# See if it's related to total async operations
```

#### 6.3 Test on Different Machine
Try running on a different machine/environment to rule out:
- Specific Python build issues
- HPC-specific configurations
- Network issues

## Expected Outcomes

By end of debugging, we should know:

1. **When**: Exactly which operation triggers the loop closure
2. **Why**: Whether it's GC, explicit cleanup, or something else
3. **Where**: Which component (httpx, openai-python, our code) closes the loop
4. **How to fix**: One of the alternative approaches from Phase 4

## Next Steps After Debugging

Based on findings, implement one of:

- **Fix A**: Global event loop singleton
- **Fix B**: Context manager approach with proper cleanup
- **Fix C**: Force synchronous mode (fallback)
- **Fix D**: Update httpx/openai-python versions
- **Fix E**: Different async library (aiohttp instead of httpx)

## Files to Create

```bash
# Create debugging script
touch scripts/debug_event_loop.py

# Run and capture output
uv run python scripts/debug_event_loop.py 2>&1 | tee debug_output.log

# Share relevant portions if needed
```

## Time Estimate

- Phase 1-3: ~1 hour (systematic isolation)
- Phase 4: ~30 min (testing alternatives)
- Phase 5-6: ~35 min (external factors)
- **Total**: ~2 hours for complete diagnosis

## Success Criteria

- [ ] Reproduce error consistently with minimal test case
- [ ] Identify exact line/operation that closes the loop
- [ ] Identify which alternative approach prevents the error
- [ ] Implement and test the fix
- [ ] Run full pipeline without errors

---

**Note**: This is a systematic scientific approach. Don't skip phases - each builds on the previous one.
