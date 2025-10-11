#!/usr/bin/env python3
"""
Validate the easy vs hard jets hypothesis through targeted experiments.

This script runs LLM classification on:
1. 100 easy jets only
2. 100 hard jets only
3. Mixed samples (50 easy + 50 hard)

Expected results if hypothesis is correct:
- Easy jets: High AUC (~0.85-0.90)
- Hard jets: Low AUC (~0.55-0.65)
- Mixed: Intermediate AUC
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_classification(classification_file: Path) -> dict:
    """Load easy/hard classification results."""
    with open(classification_file) as f:
        data = json.load(f)
    print(f"✓ Loaded classification from {classification_file}")
    print(f"  Easy jets: {data['metadata']['num_easy_jets']}")
    print(f"  Hard jets: {data['metadata']['num_hard_jets']}")
    return data


def select_jets_for_experiment(
    easy_indices: list[int],
    hard_indices: list[int],
    n_easy: int,
    n_hard: int,
    seed: int = 42
) -> tuple[np.ndarray, str]:
    """Select jets for validation experiment.

    Parameters
    ----------
    easy_indices : list
        Indices of easy jets
    hard_indices : list
        Indices of hard jets
    n_easy : int
        Number of easy jets to select
    n_hard : int
        Number of hard jets to select
    seed : int
        Random seed

    Returns
    -------
    selected_indices : np.ndarray
        Selected jet indices
    description : str
        Description of the selection
    """
    rng = np.random.RandomState(seed)

    easy_indices = np.array(easy_indices)
    hard_indices = np.array(hard_indices)

    # Sample
    if n_easy > 0 and n_easy <= len(easy_indices):
        selected_easy = rng.choice(easy_indices, size=n_easy, replace=False)
    elif n_easy > 0:
        print(f"⚠️  Requested {n_easy} easy jets but only {len(easy_indices)} available")
        selected_easy = easy_indices
    else:
        selected_easy = np.array([])

    if n_hard > 0 and n_hard <= len(hard_indices):
        selected_hard = rng.choice(hard_indices, size=n_hard, replace=False)
    elif n_hard > 0:
        print(f"⚠️  Requested {n_hard} hard jets but only {len(hard_indices)} available")
        selected_hard = hard_indices
    else:
        selected_hard = np.array([])

    # Combine and sort
    selected_indices = np.concatenate([selected_easy, selected_hard])
    selected_indices.sort()

    if n_easy > 0 and n_hard > 0:
        description = f"{n_easy}_easy_{n_hard}_hard"
    elif n_easy > 0:
        description = f"{n_easy}_easy_only"
    else:
        description = f"{n_hard}_hard_only"

    print(f"✓ Selected {len(selected_indices)} jets: {description}")

    return selected_indices, description


def create_subset_data(
    data_path: Path,
    selected_indices: np.ndarray,
    output_path: Path
) -> None:
    """Create a subset data file with selected jets."""
    data = np.load(data_path)
    # Ensure indices are integers
    selected_indices = selected_indices.astype(int)
    X_subset = data["X"][selected_indices]
    y_subset = data["y"][selected_indices]

    np.savez(output_path, X=X_subset, y=y_subset)
    print(f"✓ Created subset data file: {output_path}")
    print(f"  Shape: X={X_subset.shape}, y={y_subset.shape}")


def run_analysis(
    data_path: Path,
    output_file: Path,
    template: str,
    reasoning_effort: str,
    base_url: str,
    model_name: str,
    max_concurrent: int
) -> bool:
    """Run LLM analysis on a dataset."""
    cmd = [
        "uv", "run", "python",
        str(project_root / "scripts" / "analyze_llm_templates.py"),
        "--num_jets", str(len(np.load(data_path)["X"])),
        "--templates", template,
        "--reasoning_efforts", reasoning_effort,
        "--data_path", str(data_path),
        "--output", str(output_file),
        "--base_url", base_url,
        "--model_name", model_name,
        "--max_concurrent", str(max_concurrent),
    ]

    print(f"\nRunning analysis: {output_file.stem}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Analysis completed: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        return False


def summarize_results(results_dir: Path) -> None:
    """Summarize validation experiment results."""
    print("\n" + "="*80)
    print("VALIDATION RESULTS SUMMARY")
    print("="*80)

    # Find all result files
    result_files = sorted(results_dir.glob("validation_*.json"))

    if not result_files:
        print("No result files found")
        return

    print(f"\n{'Experiment':<30} {'N Jets':<10} {'Accuracy':<12} {'AUC':<12} {'CI Width'}")
    print("-"*80)

    summary_data = []

    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)

            # Get first result (should be only one)
            if data["results"]:
                result = data["results"][0]
                exp_name = result_file.stem.replace("validation_", "")

                # Load bootstrap if available
                bootstrap_file = result_file.with_name(result_file.stem + "_bootstrap.json")
                if bootstrap_file.exists():
                    with open(bootstrap_file) as f:
                        boot_data = json.load(f)
                    boot_result = boot_data["results"][0]["bootstrap"]
                    auc_ci_width = boot_result["auc_ci_high"] - boot_result["auc_ci_low"]
                else:
                    auc_ci_width = 0.0

                print(f"{exp_name:<30} "
                      f"{result['num_jets']:<10} "
                      f"{result['accuracy']:<12.4f} "
                      f"{result['auc']:<12.4f} "
                      f"{auc_ci_width:<12.4f}")

                summary_data.append({
                    "experiment": exp_name,
                    "num_jets": result["num_jets"],
                    "accuracy": result["accuracy"],
                    "auc": result["auc"],
                    "auc_ci_width": auc_ci_width,
                })

        except Exception as e:
            print(f"❌ Error reading {result_file}: {e}")

    print("="*80)

    # Save summary
    summary_file = results_dir / "validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n✓ Saved summary to {summary_file}")


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate easy vs hard jets hypothesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "classification_file",
        type=Path,
        help="Path to easy/hard classification JSON file",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["easy_only", "hard_only", "mixed"],
        choices=["easy_only", "hard_only", "mixed", "all"],
        help="Experiments to run",
    )

    parser.add_argument(
        "--n_jets",
        type=int,
        default=100,
        help="Number of jets per category",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model identifier",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=50,
        help="Maximum concurrent requests",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip LLM analysis (only summarize existing results)",
    )

    args = parser.parse_args()

    # Set defaults
    if args.output_dir is None:
        args.output_dir = project_root / "results" / "validation_experiments"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Expand "all" experiments
    if "all" in args.experiments:
        args.experiments = ["easy_only", "hard_only", "mixed"]

    print("="*80)
    print("EASY vs HARD JETS VALIDATION")
    print("="*80)
    print(f"Classification file: {args.classification_file}")
    print(f"Experiments:         {', '.join(args.experiments)}")
    print(f"Jets per category:   {args.n_jets}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Random seed:         {args.seed}")
    print("="*80)

    # Load classification
    classification = load_classification(args.classification_file)

    easy_indices = classification["easy_indices"]
    hard_indices = classification["hard_indices"]
    template = classification["metadata"]["template"]
    reasoning_effort = classification["metadata"]["reasoning_effort"]

    # Original data path
    data_path = project_root / "data" / "qg_jets.npz"

    if not args.skip_analysis:
        # Define experiments
        experiment_configs = []

        if "easy_only" in args.experiments:
            experiment_configs.append(("easy_only", args.n_jets, 0))

        if "hard_only" in args.experiments:
            experiment_configs.append(("hard_only", 0, args.n_jets))

        if "mixed" in args.experiments:
            n_per_cat = args.n_jets // 2
            experiment_configs.append(("mixed", n_per_cat, n_per_cat))

        print(f"\nRunning {len(experiment_configs)} experiments...")

        for exp_name, n_easy, n_hard in experiment_configs:
            print(f"\n{'='*80}")
            print(f"Experiment: {exp_name}")
            print(f"{'='*80}")

            # Select jets
            selected_indices, _ = select_jets_for_experiment(
                easy_indices, hard_indices, n_easy, n_hard, args.seed
            )

            # Create subset data file
            subset_data_path = args.output_dir / f"subset_{exp_name}.npz"
            create_subset_data(data_path, selected_indices, subset_data_path)

            # Run analysis
            output_file = args.output_dir / f"validation_{exp_name}.json"
            success = run_analysis(
                subset_data_path,
                output_file,
                template,
                reasoning_effort,
                args.base_url,
                args.model_name,
                args.max_concurrent
            )

            if not success:
                print(f"⚠️  Experiment {exp_name} failed, continuing with others...")

            # Small delay between experiments
            time.sleep(2)

    # Summarize results
    summarize_results(args.output_dir)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nInterpretation:")
    print(f"  - Easy jets should show AUC ≈ 0.85-0.90 (similar to original 100-jet sample)")
    print(f"  - Hard jets should show AUC ≈ 0.55-0.65 (near random)")
    print(f"  - Mixed sample should show intermediate performance")
    print("="*80)


if __name__ == "__main__":
    main()
