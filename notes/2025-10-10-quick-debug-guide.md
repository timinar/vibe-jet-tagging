# Quick Debug Guide - Event Loop Issue

**Status**: Error persists after 2 attempted fixes
**Next**: Systematic debugging tomorrow

## What We've Tried

1. ✗ Persistent event loop (commit 57bd8c4)
2. ✗ Lazy async client (commit 9f409cc)

Both failed - error still occurs.

## Tomorrow's Plan

### 1. Start Here (5 min)

```bash
cd /lustre/hpc/pheno/inar/vibe-jet-tagging

# Run minimal test
uv run python scripts/debug_event_loop.py
```

**Expected**: Should fail on 2nd configuration if issue reproduces.

### 2. If Test Passes (15 min)

The minimal test works but full pipeline fails → Issue is scale/concurrency related.

Test progressively:
```bash
# Increase jets: 10 → 50 → 100
uv run python scripts/debug_event_loop.py  # modify NUM_JETS in script

# Increase concurrency: 10 → 50 → 100
uv run python scripts/debug_event_loop.py  # modify max_concurrent

# More configs: 2 → 4 → 8
uv run python scripts/debug_event_loop.py  # modify configs list
```

### 3. If Test Fails (go to instrumentation)

Add debug logging (see `scripts/add_debug_logging.py` for exact code).

Run with logging:
```bash
uv run python scripts/debug_event_loop.py 2>&1 | tee debug_output.log
```

Look for:
- "Creating NEW event loop" vs "Reusing event loop"
- When `__del__` is called
- Loop closed state when `__del__` runs

### 4. Quick Fixes to Try (30 min)

#### Option A: Global Event Loop
Most likely fix - see `notes/2025-10-10-debugging-event-loop.md` Phase 4.2

#### Option B: Don't Use Async
Quick workaround:
```python
# In scripts/analyze_llm_templates.py, line 130:
predictions = clf.predict(X, verbose=False, use_async=False)  # Force sync
```

Slower, but should work.

#### Option C: Different Async Strategy
Use `asyncio.run()` each time (slower but isolated):
```python
# In local_llm_classifier.py, predict():
predictions = asyncio.run(self._predict_async(X, verbose, max_concurrent))
# Don't reuse loop
```

### 5. Nuclear Option (if all else fails)

Downgrade Python or httpx:
```bash
# Check versions
uv run python --version
uv run python -c "import httpx; print(httpx.__version__)"

# Try older httpx
uv remove httpx
uv add "httpx==0.25.0"
```

## Key Files

- `scripts/debug_event_loop.py` - Minimal reproduction test
- `scripts/add_debug_logging.py` - Instructions for instrumentation
- `notes/2025-10-10-debugging-event-loop.md` - Full detailed plan
- `src/vibe_jet_tagging/local_llm_classifier.py` - The problematic file

## Success = Finding One of These

1. **When it breaks**: Which config number (1st, 2nd, 6th?)
2. **Why it breaks**: GC timing? Loop closure? AsyncClient cleanup?
3. **How to fix**: Global loop? Sync mode? New library?

## Most Likely Root Cause (Hypothesis)

Each classifier instance creates its own event loop. When the old classifier is GC'd, its `__del__` (or httpx's cleanup) tries to use the loop, but:

- Option A: Loop is already closed
- Option B: Loop is owned by new classifier now
- Option C: Python 3.13.2 has an asyncio bug

**Test**: Use a single global event loop shared by all classifiers.

---

Budget: 2 hours maximum for debugging. If not solved, use sync mode as workaround.
