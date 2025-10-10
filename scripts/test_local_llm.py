"""Test script for LocalLLMClassifier with local OpenAI-compatible API."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier


def main():
    # Load data
    print("Loading dataset...")
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'

    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Run: uv run python scripts/download_qg_data.py -n 10000")
        return

    data = np.load(data_path)
    X = data['X']
    y = data['y']

    # Take a small sample for testing
    n_test = 10
    X_sample = X[:n_test]
    y_sample = y[:n_test]

    # Split for training (though not used in zero-shot)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sample, y_sample, test_size=0.5, random_state=42
    )

    print(f"Loaded {len(X_test)} test jets\n")

    # Test both async and sequential modes
    for use_async in [True, False]:
        mode = "ASYNC (concurrent)" if use_async else "SEQUENTIAL"
        print(f"\n{'#'*80}")
        print(f"# Testing in {mode} mode")
        print(f"{'#'*80}\n")

        # Test with medium effort only for comparison
        effort = "medium"
        print(f"{'='*80}")
        print(f"Testing with reasoning effort: {effort}")
        print(f"{'='*80}\n")

        # Create classifier
        classifier = LocalLLMClassifier(
            model_name="openai/gpt-oss-120b",
            template_name="simple_list",
            format_type="list",
            reasoning_effort=effort,
            reasoning_summary="auto",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )

        # Preview prompt for first jet (only once)
        if use_async:
            print(f"\nPreviewing prompt for first jet:")
            classifier.preview_prompt(X_test[0])

        # Fit (no-op for zero-shot)
        classifier.fit(X_train, y_train)

        # Test on 5 jets and measure time
        import time
        print(f"\n\nTesting on {len(X_test)} jets with effort={effort}, async={use_async}...\n")

        start_time = time.time()
        predictions = classifier.predict(X_test, verbose=False, use_async=use_async)
        elapsed_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"RESULTS ({mode})")
        print(f"{'='*80}")
        print(f"Predictions: {predictions}")
        print(f"True labels: {y_test.tolist()}")
        print(f"Accuracy: {np.mean(np.array(predictions) == y_test):.2%}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Average time per jet: {elapsed_time / len(X_test):.2f} seconds")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
