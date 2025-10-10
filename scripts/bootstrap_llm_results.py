#!/usr/bin/env python3
"""
Bootstrap error bars for LLM classifier results.

This script reads LLM analysis results (JSON files from analyze_llm_templates.py)
and computes bootstrapped error bars for fair comparison with baseline classifiers.

Example usage:
    python scripts/bootstrap_llm_results.py results/llm_analysis_20251010_220015.json
    python scripts/bootstrap_llm_results.py results/llm_analysis_*.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


def bootstrap_predictions(y_true, y_pred, n_bootstrap=1000):
    """
    Compute bootstrapped performance metrics from predictions.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    n_bootstrap : int
        Number of bootstrap samples (default: 1000)

    Returns
    -------
    dict
        Dictionary with mean, std, and confidence intervals for accuracy and AUC
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)

    accuracies = []
    aucs = []

    for _ in range(n_bootstrap):
        # Bootstrap resample indices
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Compute metrics
        acc = accuracy_score(y_true_boot, y_pred_boot)
        try:
            auc = roc_auc_score(y_true_boot, y_pred_boot)
        except ValueError:
            # If all same class in bootstrap sample, skip or use 0.5
            auc = 0.5

        accuracies.append(acc)
        aucs.append(auc)

    accuracies = np.array(accuracies)
    aucs = np.array(aucs)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "accuracy_ci_low": float(np.percentile(accuracies, 2.5)),
        "accuracy_ci_high": float(np.percentile(accuracies, 97.5)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_ci_low": float(np.percentile(aucs, 2.5)),
        "auc_ci_high": float(np.percentile(aucs, 97.5)),
    }


def process_results_file(results_path: Path, n_bootstrap: int = 1000) -> dict:
    """
    Process a single LLM results JSON file and add bootstrap error bars.

    Parameters
    ----------
    results_path : Path
        Path to LLM results JSON file
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    dict
        Results with added bootstrap statistics
    """
    print(f"\n{'='*70}")
    print(f"Processing: {results_path.name}")
    print(f"{'='*70}")

    # Load results
    with open(results_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    results = data["results"]

    print(f"\nMetadata:")
    print(f"  Model: {metadata['model_name']}")
    print(f"  Number of jets: {metadata['num_jets']}")
    print(f"  Configurations: {len(results)}")

    # Bootstrap each configuration
    print(f"\nBootstrapping with {n_bootstrap} samples...")
    bootstrapped_results = []

    for result in tqdm(results, desc="Configurations"):
        y_true = np.array(result["true_labels"])
        y_pred = np.array(result["predictions"])

        # Compute bootstrap statistics
        boot_stats = bootstrap_predictions(y_true, y_pred, n_bootstrap=n_bootstrap)

        # Add bootstrap stats to result
        result_with_bootstrap = result.copy()
        result_with_bootstrap["bootstrap"] = boot_stats

        bootstrapped_results.append(result_with_bootstrap)

    # Create output data
    output_data = {
        "metadata": metadata,
        "metadata_bootstrap": {
            "n_bootstrap": n_bootstrap,
            "bootstrap_method": "resampling with replacement",
            "ci_level": 0.95,
        },
        "results": bootstrapped_results,
    }

    return output_data


def print_summary(data: dict):
    """Print summary table with bootstrap error bars."""
    results = data["results"]
    n_bootstrap = data["metadata_bootstrap"]["n_bootstrap"]

    print(f"\n{'='*100}")
    print("RESULTS WITH BOOTSTRAP ERROR BARS")
    print(f"{'='*100}")
    print(
        f"{'Template':<25} {'Effort':<10} {'Accuracy':<25} {'AUC':<25}"
    )
    print("-" * 100)

    for result in results:
        boot = result["bootstrap"]
        template = result["template"]
        effort = result["reasoning_effort"]

        acc_str = (
            f"{boot['accuracy_mean']:.3f} ± {boot['accuracy_std']:.3f} "
            f"[{boot['accuracy_ci_low']:.3f}, {boot['accuracy_ci_high']:.3f}]"
        )
        auc_str = (
            f"{boot['auc_mean']:.3f} ± {boot['auc_std']:.3f} "
            f"[{boot['auc_ci_low']:.3f}, {boot['auc_ci_high']:.3f}]"
        )

        print(f"{template:<25} {effort:<10} {acc_str:<25} {auc_str:<25}")

    print("=" * 100)
    print(f"Note: Error bars from {n_bootstrap} bootstrap resamples")
    print("      Format: mean ± std [95% CI]")


def main():
    """Main script."""
    parser = argparse.ArgumentParser(
        description="Add bootstrap error bars to LLM classifier results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_files",
        nargs="+",
        type=Path,
        help="Path(s) to LLM results JSON file(s)",
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples",
    )

    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_bootstrap",
        help="Suffix to add to output filenames",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LLM RESULTS BOOTSTRAP ANALYSIS")
    print("=" * 70)
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Input files: {len(args.results_files)}")

    all_results = []

    for results_path in args.results_files:
        if not results_path.exists():
            print(f"\n❌ File not found: {results_path}")
            continue

        # Process file
        try:
            output_data = process_results_file(results_path, n_bootstrap=args.n_bootstrap)

            # Save bootstrapped results
            output_path = results_path.parent / f"{results_path.stem}{args.output_suffix}.json"
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\n✓ Saved: {output_path}")

            # Print summary
            print_summary(output_data)

            all_results.append((results_path.name, output_data))

        except Exception as e:
            print(f"\n❌ Error processing {results_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Overall summary if multiple files
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY: ALL FILES")
        print(f"{'='*70}")
        for filename, data in all_results:
            print(f"\n{filename}:")
            for result in data["results"]:
                boot = result["bootstrap"]
                print(
                    f"  {result['template']:<25} ({result['reasoning_effort']:<6}): "
                    f"AUC = {boot['auc_mean']:.3f} ± {boot['auc_std']:.3f}"
                )

    print(f"\n{'='*70}")
    print("✓ Bootstrap analysis complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
