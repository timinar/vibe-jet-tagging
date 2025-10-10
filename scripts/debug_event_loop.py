#!/usr/bin/env python3
"""
Minimal test case to debug event loop closure issue.

This script reproduces the "Event loop is closed" error with minimal overhead
to help identify the root cause.

Usage:
    uv run python scripts/debug_event_loop.py

With debug output:
    uv run python scripts/debug_event_loop.py 2>&1 | tee debug_output.log
"""

import gc
import sys
from pathlib import Path

import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier


def main():
    print("=" * 80)
    print("EVENT LOOP DEBUGGING TEST")
    print("=" * 80)

    # Load minimal data
    data_path = project_root / "data" / "qg_jets.npz"
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    X = data["X"][:10]  # Just 10 jets for fast testing
    y = data["y"][:10]
    print(f"✓ Loaded {len(X)} jets")

    # Test configurations - start with just 2 identical configs
    configs = [
        ("simple_list", "low"),
        ("simple_list", "low"),  # Same config twice to isolate issue
    ]

    print(f"\nTesting {len(configs)} sequential configurations...")
    print("=" * 80)

    for i, (template, effort) in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(configs)}] Configuration: {template} | {effort}")
        print(f"{'='*80}")

        # Create classifier
        print(f"\n1. Creating classifier...")
        clf = LocalLLMClassifier(
            model_name="openai/gpt-oss-120b",
            template_name=template,
            reasoning_effort=effort,
        )
        print(f"   ✓ Classifier created (id={id(clf)})")

        # Fit
        clf.fit(X, y)
        print(f"   ✓ Classifier fitted")

        # Check state before predict
        print(f"\n2. State before predict():")
        print(f"   - Event loop: {clf._event_loop}")
        print(f"   - Async client: {clf._async_client}")

        # Make predictions
        print(f"\n3. Running predict() with 10 jets (max_concurrent=10)...")
        try:
            predictions = clf.predict(X, verbose=False, use_async=True, max_concurrent=10)
            print(f"   ✓ Predictions complete: {len(predictions)} made")

        except Exception as e:
            print(f"\n   ❌ ERROR during predict():")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n   Test FAILED at configuration {i}/{len(configs)}")
            return 1

        # Check state after predict
        print(f"\n4. State after predict():")
        print(f"   - Event loop: {clf._event_loop}")
        if clf._event_loop:
            print(f"   - Event loop closed: {clf._event_loop.is_closed()}")
        print(f"   - Async client: {clf._async_client}")

        # Explicitly delete and garbage collect
        print(f"\n5. Cleanup:")
        print(f"   - Deleting classifier...")
        clf_id = id(clf)
        del clf

        print(f"   - Running garbage collection...")
        gc.collect()
        print(f"   ✓ Cleanup complete (former id={clf_id})")

        # Small delay to let async cleanup finish
        import time
        time.sleep(0.5)

    print(f"\n{'='*80}")
    print("TEST COMPLETE - All configurations succeeded!")
    print(f"{'='*80}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
