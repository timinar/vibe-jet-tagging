#!/usr/bin/env python3
"""
Analyze LocalLLMClassifier performance across templates and reasoning efforts.

This script evaluates the LocalLLMClassifier on quark/gluon jet tagging with:
- Multiple prompt templates
- Different reasoning effort levels
- Comprehensive metrics (AUC, accuracy, token usage, generation time)

Results are saved to a JSON file for further analysis and plotting.

Example usage:
    # Basic run with 10 jets
    python scripts/analyze_llm_templates.py

    # Run with custom settings
    python scripts/analyze_llm_templates.py --num_jets 100 \
        --reasoning_efforts low medium high \
        --templates "simple_list" "with_summary_stats"

    # For screen/tmux
    screen -S llm_analysis
    python scripts/analyze_llm_templates.py --num_jets 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Add src to path if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vibe_jet_tagging.local_llm_classifier import LocalLLMClassifier


def load_data(
    data_path: Path,
    num_jets: int,
    random_sample: bool = False,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load jet data and labels.

    Parameters
    ----------
    data_path : Path
        Path to the data file
    num_jets : int
        Number of jets to load
    random_sample : bool
        If True, randomly sample jets; otherwise use sequential indices
    seed : int
        Random seed for reproducibility

    Returns
    -------
    X : np.ndarray
        Jet data (num_jets, n_particles, 4)
    y : np.ndarray
        Labels (num_jets,)
    indices : np.ndarray
        Jet indices in the original dataset
    """
    data = np.load(data_path)
    total_jets = len(data["X"])

    if random_sample:
        rng = np.random.RandomState(seed)
        indices = rng.choice(total_jets, size=num_jets, replace=False)
        indices.sort()  # Sort for reproducibility
        print(f"✓ Randomly sampled {num_jets} jets (seed={seed})")
    else:
        indices = np.arange(num_jets)
        print(f"✓ Using sequential jets [0:{num_jets}]")

    X = data["X"][indices]
    y = data["y"][indices]

    print(f"  - Quark jets: {(y == 1).sum()}")
    print(f"  - Gluon jets: {(y == 0).sum()}")
    print(f"  - Index range: [{indices.min()}, {indices.max()}]")

    return X, y, indices


def evaluate_configuration(
    template_name: str,
    reasoning_effort: str,
    X: np.ndarray,
    y: np.ndarray,
    jet_indices: np.ndarray,
    base_url: str,
    api_key: str,
    templates_dir: Path,
    model_name: str = "openai/gpt-oss-120b",
    max_concurrent: int = 50,
) -> dict[str, Any]:
    """Evaluate a single template + reasoning effort configuration.

    Parameters
    ----------
    template_name : str
        Name of the prompt template
    reasoning_effort : str
        Reasoning effort level: "low", "medium", or "high"
    X : np.ndarray
        Jet data
    y : np.ndarray
        True labels
    base_url : str
        Base URL for the API
    api_key : str
        API key
    templates_dir : Path
        Directory containing templates
    model_name : str
        Model identifier
    max_concurrent : int
        Maximum number of concurrent requests

    Returns
    -------
    results : dict
        Dictionary containing metrics and predictions
    """
    print(f"\n{'='*70}")
    print(f"Template: {template_name} | Reasoning: {reasoning_effort}")
    print(f"{'='*70}")

    # Initialize classifier
    clf = LocalLLMClassifier(
        model_name=model_name,
        template_name=template_name,
        format_type="list",
        reasoning_effort=reasoning_effort,
        base_url=base_url,
        api_key=api_key,
        templates_dir=str(templates_dir),
    )

    clf.fit([], [])

    # Make predictions
    print(f"Processing {len(X)} jets (async mode, max {max_concurrent} concurrent)...")
    start_time = time.time()
    predictions = clf.predict(X, verbose=False, use_async=True, max_concurrent=max_concurrent)
    elapsed_time = time.time() - start_time

    # Calculate metrics
    predictions = np.array(predictions)
    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, predictions)

    # Collect results
    results = {
        "template": template_name,
        "reasoning_effort": reasoning_effort,
        "accuracy": float(accuracy),
        "auc": float(auc),
        "num_jets": int(len(X)),
        "total_time": float(elapsed_time),
        "avg_time_per_jet": float(elapsed_time / len(X)),
        "total_generation_time": float(clf.total_generation_time),
        "prompt_tokens": int(clf.total_prompt_tokens),
        "completion_tokens": int(clf.total_completion_tokens),
        "reasoning_tokens": int(clf.total_reasoning_tokens),
        "total_tokens": int(
            clf.total_prompt_tokens
            + clf.total_completion_tokens
            + clf.total_reasoning_tokens
        ),
        "predictions": predictions.tolist(),
        "true_labels": y.tolist(),
        "jet_indices": jet_indices.tolist(),
    }

    # Print summary
    print(f"\n✓ Results:")
    print(f"  Accuracy:              {accuracy:.4f}")
    print(f"  AUC:                   {auc:.4f}")
    print(f"  Total time:            {elapsed_time:.2f}s")
    print(f"  Avg time per jet:      {elapsed_time/len(X):.3f}s")
    print(f"  Generation time:       {clf.total_generation_time:.2f}s")
    print(f"  Total tokens:          {results['total_tokens']:,}")
    print(f"  Reasoning tokens (est): {clf.total_reasoning_tokens:,}")

    return results


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze LLM classifier performance across templates and reasoning efforts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_jets",
        type=int,
        default=10,
        help="Number of jets to evaluate",
    )

    parser.add_argument(
        "--reasoning_efforts",
        nargs="+",
        default=["low", "medium"],
        choices=["low", "medium", "high"],
        help="Reasoning effort levels to test",
    )

    parser.add_argument(
        "--templates",
        nargs="+",
        default=[
            "simple_list",
            "with_summary_stats",
            "with_optimal_cut",
            "with_engineered_features",
        ],
        help="Templates to evaluate",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the OpenAI-compatible API",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key (use EMPTY for local servers)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model identifier",
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to data file (default: PROJECT_ROOT/data/qg_jets.npz)",
    )

    parser.add_argument(
        "--templates_dir",
        type=Path,
        default=None,
        help="Path to templates directory (default: PROJECT_ROOT/templates)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path (default: results/llm_analysis_TIMESTAMP.json)",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=50,
        help="Maximum number of concurrent requests (default: 50)",
    )

    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Randomly sample jets instead of sequential",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jet sampling (default: 42)",
    )

    args = parser.parse_args()

    # Set default paths
    if args.data_path is None:
        args.data_path = project_root / "data" / "qg_jets.npz"
    if args.templates_dir is None:
        args.templates_dir = project_root / "templates"
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "results"
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"llm_analysis_{timestamp}.json"

    # Print configuration
    print("="*80)
    print("LLM CLASSIFIER ANALYSIS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of jets:      {args.num_jets}")
    print(f"  Templates:           {', '.join(args.templates)}")
    print(f"  Reasoning efforts:   {', '.join(args.reasoning_efforts)}")
    print(f"  Model:               {args.model_name}")
    print(f"  Base URL:            {args.base_url}")
    print(f"  Max concurrent:      {args.max_concurrent}")
    print(f"  Data path:           {args.data_path}")
    print(f"  Templates dir:       {args.templates_dir}")
    print(f"  Output file:         {args.output}")
    print(f"\nTotal configurations: {len(args.templates) * len(args.reasoning_efforts)}")
    print(f"Total API calls:      {len(args.templates) * len(args.reasoning_efforts) * args.num_jets}")
    print("="*80)

    # Check server connection
    try:
        import requests
        response = requests.get(f"{args.base_url.replace('/v1', '')}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"\n✓ Connected to server")
            print(f"  Available models: {[m['id'] for m in models.get('data', [])]}")
        else:
            print(f"\n⚠️  Server responded but with status {response.status_code}")
    except Exception as e:
        print(f"\n❌ Could not connect to server at {args.base_url}")
        print(f"   Error: {e}")
        print(f"\n   Please start the server first:")
        print(f"   vllm serve {args.model_name} --port 8000")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    X, y, jet_indices = load_data(args.data_path, args.num_jets, args.random_sample, args.seed)

    # Run all configurations
    all_results = []
    total_configs = len(args.templates) * len(args.reasoning_efforts)
    config_num = 0

    start_total = time.time()

    for template in args.templates:
        for effort in args.reasoning_efforts:
            config_num += 1
            print(f"\n[{config_num}/{total_configs}] Starting configuration...")

            try:
                results = evaluate_configuration(
                    template_name=template,
                    reasoning_effort=effort,
                    X=X,
                    y=y,
                    jet_indices=jet_indices,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    templates_dir=args.templates_dir,
                    model_name=args.model_name,
                    max_concurrent=args.max_concurrent,
                )
                all_results.append(results)

            except Exception as e:
                print(f"\n❌ Error in configuration: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_total

    # Save results
    output_data = {
        "metadata": {
            "num_jets": args.num_jets,
            "templates": args.templates,
            "reasoning_efforts": args.reasoning_efforts,
            "model_name": args.model_name,
            "base_url": args.base_url,
            "data_path": str(args.data_path),
            "templates_dir": str(args.templates_dir),
            "random_sample": args.random_sample,
            "seed": args.seed,
            "total_elapsed_time": float(total_elapsed),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": all_results,
    }

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Total configurations:  {len(all_results)}/{total_configs}")
    print(f"Total elapsed time:    {total_elapsed:.2f}s")
    print(f"\nSaving results to: {args.output}")

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Template':<30} {'Effort':<10} {'Accuracy':<10} {'AUC':<10} {'Tokens':<12}")
    print("-"*80)

    for result in all_results:
        print(
            f"{result['template']:<30} "
            f"{result['reasoning_effort']:<10} "
            f"{result['accuracy']:<10.4f} "
            f"{result['auc']:<10.4f} "
            f"{result['total_tokens']:<12,}"
        )

    print("="*80)

    # Find best configuration
    if all_results:
        best_result = max(all_results, key=lambda x: x["auc"])
        print(f"\n🏆 Best configuration:")
        print(f"   Template: {best_result['template']}")
        print(f"   Reasoning effort: {best_result['reasoning_effort']}")
        print(f"   AUC: {best_result['auc']:.4f}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")

    print(f"\n💡 Next step: Run plotting script to visualize results")
    print(f"   python scripts/plot_llm_analysis.py {args.output}")


if __name__ == "__main__":
    main()
