#!/usr/bin/env python3
"""
Analyze easy vs hard jets to understand performance degradation.

This script:
1. Loads LLM analysis results with jet indices
2. Classifies jets as easy (correct) vs hard (incorrect)
3. Computes kinematic and substructure properties
4. Identifies distinguishing characteristics
5. Saves classification and analysis results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_analysis_results(results_path: Path) -> dict[str, Any]:
    """Load LLM analysis results."""
    with open(results_path) as f:
        data = json.load(f)
    print(f"✓ Loaded results from {results_path}")
    print(f"  - Number of configurations: {len(data['results'])}")
    return data


def classify_jets_easy_hard(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    jet_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Classify jets as easy (correct) or hard (incorrect).

    Returns
    -------
    easy_indices : np.ndarray
        Indices of jets classified correctly
    hard_indices : np.ndarray
        Indices of jets classified incorrectly
    """
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    jet_indices = np.array(jet_indices)

    correct = predictions == true_labels

    easy_indices = jet_indices[correct]
    hard_indices = jet_indices[~correct]

    print(f"\nJet Classification:")
    print(f"  Easy jets (correct): {len(easy_indices)} ({len(easy_indices)/len(jet_indices)*100:.1f}%)")
    print(f"  Hard jets (incorrect): {len(hard_indices)} ({len(hard_indices)/len(jet_indices)*100:.1f}%)")

    return easy_indices, hard_indices


def compute_jet_properties(X: np.ndarray) -> dict[str, np.ndarray]:
    """Compute kinematic and substructure properties for jets.

    Parameters
    ----------
    X : np.ndarray
        Jet data (num_jets, n_particles, 4) where features are (px, py, pz, E)

    Returns
    -------
    properties : dict
        Dictionary of property arrays
    """
    print("\nComputing jet properties...")

    # Constituent-level info (non-zero entries)
    # Assuming zero padding for empty constituent slots
    has_particle = (X[..., 3] > 0)  # E > 0 indicates real particle

    # Number of constituents
    n_constituents = has_particle.sum(axis=1)

    # Compute derived quantities
    pt = np.sqrt(X[..., 0]**2 + X[..., 1]**2)  # Transverse momentum
    eta = np.arctanh(X[..., 2] / np.sqrt(X[..., 0]**2 + X[..., 1]**2 + X[..., 2]**2 + 1e-10))
    phi = np.arctan2(X[..., 1], X[..., 0])

    # Mask out padding
    pt_masked = np.where(has_particle, pt, 0)
    eta_masked = np.where(has_particle, eta, 0)
    phi_masked = np.where(has_particle, phi, 0)

    # Jet-level properties
    jet_pt = pt_masked.sum(axis=1)
    jet_mass = np.sqrt(np.maximum(0, X[..., 3].sum(axis=1)**2 -
                                    X[..., 0].sum(axis=1)**2 -
                                    X[..., 1].sum(axis=1)**2 -
                                    X[..., 2].sum(axis=1)**2))

    # Leading particle pT fraction
    leading_pt_frac = np.where(jet_pt > 0, pt_masked.max(axis=1) / jet_pt, 0)

    # pT dispersion (among constituents)
    pt_mean = np.where(n_constituents > 0, pt_masked.sum(axis=1) / n_constituents, 0)
    pt_var = np.where(
        n_constituents > 0,
        ((pt_masked - pt_mean[:, None])**2 * has_particle).sum(axis=1) / n_constituents,
        0
    )
    pt_dispersion = np.sqrt(pt_var)

    # Angular properties
    # Center: pT-weighted average
    eta_center = np.where(jet_pt > 0, (eta_masked * pt_masked).sum(axis=1) / jet_pt, 0)
    phi_center = np.where(jet_pt > 0, (phi_masked * pt_masked).sum(axis=1) / jet_pt, 0)

    # ΔR from jet center
    delta_eta = eta_masked - eta_center[:, None]
    delta_phi = phi_masked - phi_center[:, None]
    # Handle phi wrapping
    delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

    # Angular width (pT-weighted average ΔR)
    angular_width = np.where(
        jet_pt > 0,
        (delta_r * pt_masked).sum(axis=1) / jet_pt,
        0
    )

    # Girth: Σ(pT_i × ΔR_i) / Σ(pT_i)
    girth = angular_width  # Same definition

    # pT^D: Σ(pT_i²) / (Σ pT_i)²
    pt_squared_sum = (pt_masked**2).sum(axis=1)
    ptD = np.where(jet_pt > 0, pt_squared_sum / jet_pt**2, 0)

    properties = {
        "n_constituents": n_constituents,
        "jet_pt": jet_pt,
        "jet_mass": jet_mass,
        "leading_pt_frac": leading_pt_frac,
        "pt_dispersion": pt_dispersion,
        "angular_width": angular_width,
        "girth": girth,
        "ptD": ptD,
    }

    print(f"✓ Computed {len(properties)} jet properties for {len(X)} jets")

    return properties


def compare_properties(
    easy_props: dict[str, np.ndarray],
    hard_props: dict[str, np.ndarray],
    output_dir: Path
) -> dict[str, dict]:
    """Compare properties between easy and hard jets.

    Returns
    -------
    stats_results : dict
        Statistical test results for each property
    """
    print("\nComparing easy vs hard jet properties...")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_results = {}

    # Create comparison plots
    n_props = len(easy_props)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (prop_name, easy_vals) in enumerate(easy_props.items()):
        hard_vals = hard_props[prop_name]

        # Statistical test (Mann-Whitney U test)
        statistic, pvalue = stats.mannwhitneyu(easy_vals, hard_vals, alternative='two-sided')

        # Effect size (Cohen's d)
        mean_easy, std_easy = np.mean(easy_vals), np.std(easy_vals)
        mean_hard, std_hard = np.mean(hard_vals), np.std(hard_vals)
        pooled_std = np.sqrt((std_easy**2 + std_hard**2) / 2)
        cohens_d = (mean_easy - mean_hard) / pooled_std if pooled_std > 0 else 0

        stats_results[prop_name] = {
            "easy_mean": float(mean_easy),
            "easy_std": float(std_easy),
            "hard_mean": float(mean_hard),
            "hard_std": float(std_hard),
            "mann_whitney_u": float(statistic),
            "p_value": float(pvalue),
            "cohens_d": float(cohens_d),
            "significant": bool(pvalue < 0.05),
        }

        # Plot
        ax = axes[idx]
        bins = np.linspace(
            min(easy_vals.min(), hard_vals.min()),
            max(easy_vals.max(), hard_vals.max()),
            50
        )

        ax.hist(easy_vals, bins=bins, alpha=0.5, label='Easy', density=True, color='blue')
        ax.hist(hard_vals, bins=bins, alpha=0.5, label='Hard', density=True, color='red')
        ax.set_xlabel(prop_name.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()

        # Add statistics
        sig_marker = '***' if pvalue < 0.001 else ('**' if pvalue < 0.01 else ('*' if pvalue < 0.05 else 'ns'))
        ax.text(0.95, 0.95, f"p={pvalue:.3e}\nd={cohens_d:.2f} {sig_marker}",
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "easy_vs_hard_properties.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "easy_vs_hard_properties.pdf", bbox_inches='tight')
    print(f"✓ Saved property comparison plots to {output_dir}")

    # Print summary
    print("\nStatistical Comparison Summary:")
    print("="*90)
    print(f"{'Property':<20} {'Easy Mean':<12} {'Hard Mean':<12} {'Cohen d':<10} {'p-value':<12} {'Sig'}")
    print("-"*90)

    for prop_name, result in stats_results.items():
        sig_marker = '***' if result['p_value'] < 0.001 else (
            '**' if result['p_value'] < 0.01 else (
            '*' if result['p_value'] < 0.05 else 'ns'))

        print(f"{prop_name:<20} "
              f"{result['easy_mean']:<12.4f} "
              f"{result['hard_mean']:<12.4f} "
              f"{result['cohens_d']:<10.2f} "
              f"{result['p_value']:<12.3e} "
              f"{sig_marker}")

    print("="*90)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")

    return stats_results


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze easy vs hard jets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to LLM analysis results JSON file (with jet indices)",
    )

    parser.add_argument(
        "--template",
        type=str,
        default="with_optimal_cut",
        help="Template to analyze",
    )

    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        help="Reasoning effort to analyze",
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to jet data file",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    # Set defaults
    if args.data_path is None:
        args.data_path = project_root / "data" / "qg_jets.npz"
    if args.output_dir is None:
        args.output_dir = project_root / "results" / "easy_hard_analysis"

    print("="*80)
    print("EASY vs HARD JETS ANALYSIS")
    print("="*80)
    print(f"Results file:    {args.results_file}")
    print(f"Template:        {args.template}")
    print(f"Reasoning:       {args.reasoning_effort}")
    print(f"Data path:       {args.data_path}")
    print(f"Output dir:      {args.output_dir}")
    print("="*80)

    # Load analysis results
    data = load_analysis_results(args.results_file)

    # Find the specific configuration
    result = None
    for r in data["results"]:
        if r["template"] == args.template and r["reasoning_effort"] == args.reasoning_effort:
            result = r
            break

    if result is None:
        print(f"\n❌ Could not find results for template={args.template}, reasoning={args.reasoning_effort}")
        print(f"Available configurations:")
        for r in data["results"]:
            print(f"  - {r['template']}, {r['reasoning_effort']}")
        sys.exit(1)

    print(f"\n✓ Found configuration:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  AUC: {result['auc']:.4f}")
    print(f"  Num jets: {result['num_jets']}")

    # Classify jets
    easy_indices, hard_indices = classify_jets_easy_hard(
        result["predictions"],
        result["true_labels"],
        result["jet_indices"]
    )

    # Load jet data
    print(f"\nLoading jet data from {args.data_path}...")
    data_npz = np.load(args.data_path)
    X_all = data_npz["X"]
    y_all = data_npz["y"]

    # Extract easy and hard jets
    X_easy = X_all[easy_indices]
    y_easy = y_all[easy_indices]
    X_hard = X_all[hard_indices]
    y_hard = y_all[hard_indices]

    print(f"✓ Loaded jet data")
    print(f"  Easy jets: {len(X_easy)} (quark: {(y_easy==1).sum()}, gluon: {(y_easy==0).sum()})")
    print(f"  Hard jets: {len(X_hard)} (quark: {(y_hard==1).sum()}, gluon: {(y_hard==0).sum()})")

    # Compute properties
    easy_props = compute_jet_properties(X_easy)
    hard_props = compute_jet_properties(X_hard)

    # Compare properties
    stats_results = compare_properties(easy_props, hard_props, args.output_dir)

    # Save results
    output_data = {
        "metadata": {
            "results_file": str(args.results_file),
            "template": args.template,
            "reasoning_effort": args.reasoning_effort,
            "num_easy_jets": int(len(easy_indices)),
            "num_hard_jets": int(len(hard_indices)),
            "accuracy": float(result["accuracy"]),
            "auc": float(result["auc"]),
        },
        "easy_indices": easy_indices.tolist(),
        "hard_indices": hard_indices.tolist(),
        "easy_labels": y_easy.tolist(),
        "hard_labels": y_hard.tolist(),
        "statistical_comparison": stats_results,
    }

    output_file = args.output_dir / "easy_hard_classification.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved classification results to {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Review property distributions in {args.output_dir}/easy_vs_hard_properties.png")
    print(f"  2. Run targeted validation:")
    print(f"     python scripts/validate_easy_hard_hypothesis.py {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
