#!/usr/bin/env python3
"""
Advanced property analysis for easy vs hard jets.

Since basic kinematics show no difference, we explore:
1. Higher-order substructure
2. Constituent-level details
3. Energy flow patterns
4. Spatial distributions
5. Correlations and combinations
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def compute_advanced_properties(X: np.ndarray) -> dict[str, np.ndarray]:
    """Compute advanced jet properties.

    Parameters
    ----------
    X : np.ndarray
        Jet data (num_jets, n_particles, 4) where features are (px, py, pz, E)

    Returns
    -------
    properties : dict
        Dictionary of property arrays
    """
    print("\nComputing advanced jet properties...")

    # Constituent-level info
    has_particle = (X[..., 3] > 0)  # E > 0 indicates real particle
    n_constituents = has_particle.sum(axis=1)

    # Compute derived quantities
    pt = np.sqrt(X[..., 0]**2 + X[..., 1]**2)
    eta = np.arctanh(X[..., 2] / np.sqrt(X[..., 0]**2 + X[..., 1]**2 + X[..., 2]**2 + 1e-10))
    phi = np.arctan2(X[..., 1], X[..., 0])
    E = X[..., 3]

    # Mask out padding
    pt_masked = np.where(has_particle, pt, 0)
    eta_masked = np.where(has_particle, eta, 0)
    phi_masked = np.where(has_particle, phi, 0)
    E_masked = np.where(has_particle, E, 0)

    # Jet-level
    jet_pt = pt_masked.sum(axis=1)
    jet_E = E_masked.sum(axis=1)

    # pT-weighted centers
    eta_center = np.where(jet_pt > 0, (eta_masked * pt_masked).sum(axis=1) / jet_pt, 0)
    phi_center = np.where(jet_pt > 0, (phi_masked * pt_masked).sum(axis=1) / jet_pt, 0)

    # ΔR from center
    delta_eta = eta_masked - eta_center[:, None]
    delta_phi = phi_masked - phi_center[:, None]
    delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)
    delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

    properties = {}

    # ============= 1. HIGHER-ORDER MOMENTS =============

    # pT moments
    pt_skewness = compute_masked_skewness(pt_masked, has_particle)
    pt_kurtosis = compute_masked_kurtosis(pt_masked, has_particle)
    properties["pt_skewness"] = pt_skewness
    properties["pt_kurtosis"] = pt_kurtosis

    # Energy moments
    E_skewness = compute_masked_skewness(E_masked, has_particle)
    E_kurtosis = compute_masked_kurtosis(E_masked, has_particle)
    properties["E_skewness"] = E_skewness
    properties["E_kurtosis"] = E_kurtosis

    # ΔR moments
    dr_skewness = compute_masked_skewness(delta_r, has_particle)
    dr_kurtosis = compute_masked_kurtosis(delta_r, has_particle)
    properties["dr_skewness"] = dr_skewness
    properties["dr_kurtosis"] = dr_kurtosis

    # ============= 2. MULTIPLICITY ANALYSIS =============

    # Soft particles (pT < 10 GeV)
    soft_particles = (pt_masked > 0) & (pt_masked < 10)
    n_soft = soft_particles.sum(axis=1)
    properties["n_soft_particles"] = n_soft
    properties["soft_fraction"] = np.where(n_constituents > 0, n_soft / n_constituents, 0)

    # Hard particles (pT > 50 GeV)
    hard_particles = pt_masked > 50
    n_hard = hard_particles.sum(axis=1)
    properties["n_hard_particles"] = n_hard

    # ============= 3. ENERGY FLOW PATTERNS =============

    # Radial energy flow (binned by ΔR)
    dr_bins = [0, 0.1, 0.2, 0.3, 0.4]
    for i in range(len(dr_bins)-1):
        r_min, r_max = dr_bins[i], dr_bins[i+1]
        in_bin = (delta_r >= r_min) & (delta_r < r_max) & has_particle
        energy_in_bin = np.where(in_bin, E_masked, 0).sum(axis=1)
        properties[f"E_frac_r{int(r_min*10)}{int(r_max*10)}"] = np.where(
            jet_E > 0, energy_in_bin / jet_E, 0
        )

    # ============= 4. SPATIAL DISTRIBUTION =============

    # Eccentricity (asymmetry of particle distribution)
    # Using pT-weighted η-φ covariance
    cov_eta_eta = np.where(
        jet_pt > 0,
        ((delta_eta**2) * pt_masked * has_particle).sum(axis=1) / jet_pt,
        0
    )
    cov_phi_phi = np.where(
        jet_pt > 0,
        ((delta_phi**2) * pt_masked * has_particle).sum(axis=1) / jet_pt,
        0
    )
    cov_eta_phi = np.where(
        jet_pt > 0,
        ((delta_eta * delta_phi) * pt_masked * has_particle).sum(axis=1) / jet_pt,
        0
    )

    # Eigenvalues of covariance matrix
    discriminant = np.sqrt(
        (cov_eta_eta - cov_phi_phi)**2 + 4*cov_eta_phi**2
    )
    lambda_1 = 0.5 * (cov_eta_eta + cov_phi_phi + discriminant)
    lambda_2 = 0.5 * (cov_eta_eta + cov_phi_phi - discriminant)

    eccentricity = np.where(
        lambda_1 > 0,
        1 - lambda_2 / lambda_1,
        0
    )
    properties["eccentricity"] = eccentricity

    # ============= 5. N-SUBJETTINESS RATIOS =============

    # Simplified 2-subjettiness / 1-subjettiness
    # 1-subjettiness: already computed (girth)
    tau_1 = np.where(
        jet_pt > 0,
        (delta_r * pt_masked * has_particle).sum(axis=1) / jet_pt,
        0
    )

    # 2-subjettiness: find 2 sub-centers by k-means approximation
    # Simplified: use leading 2 particles as axes
    pt_sorted_idx = np.argsort(-pt_masked, axis=1)

    tau_2_list = []
    for jet_idx in range(len(X)):
        if n_constituents[jet_idx] < 2:
            tau_2_list.append(0.0)
            continue

        # Get 2 leading particles as axes
        idx1, idx2 = pt_sorted_idx[jet_idx, 0], pt_sorted_idx[jet_idx, 1]

        eta1, phi1 = eta_masked[jet_idx, idx1], phi_masked[jet_idx, idx1]
        eta2, phi2 = eta_masked[jet_idx, idx2], phi_masked[jet_idx, idx2]

        # For each particle, distance to nearest axis
        d_eta1 = eta_masked[jet_idx] - eta1
        d_phi1 = phi_masked[jet_idx] - phi1
        d_phi1 = np.where(d_phi1 > np.pi, d_phi1 - 2*np.pi, d_phi1)
        d_phi1 = np.where(d_phi1 < -np.pi, d_phi1 + 2*np.pi, d_phi1)
        dr1 = np.sqrt(d_eta1**2 + d_phi1**2)

        d_eta2 = eta_masked[jet_idx] - eta2
        d_phi2 = phi_masked[jet_idx] - phi2
        d_phi2 = np.where(d_phi2 > np.pi, d_phi2 - 2*np.pi, d_phi2)
        d_phi2 = np.where(d_phi2 < -np.pi, d_phi2 + 2*np.pi, d_phi2)
        dr2 = np.sqrt(d_eta2**2 + d_phi2**2)

        min_dr = np.minimum(dr1, dr2)
        tau_2 = (min_dr * pt_masked[jet_idx] * has_particle[jet_idx]).sum() / jet_pt[jet_idx]
        tau_2_list.append(tau_2)

    tau_2 = np.array(tau_2_list)
    tau_21 = np.where(tau_1 > 0, tau_2 / tau_1, 0)
    properties["tau_21"] = tau_21

    # ============= 6. CONSTITUENT CORRELATIONS =============

    # pT vs ΔR correlation
    pt_dr_corr = compute_masked_correlation(
        pt_masked, delta_r, has_particle
    )
    properties["pt_dr_correlation"] = pt_dr_corr

    # Energy vs ΔR correlation
    E_dr_corr = compute_masked_correlation(
        E_masked, delta_r, has_particle
    )
    properties["E_dr_correlation"] = E_dr_corr

    # ============= 7. NUMERIC PRECISION INDICATORS =============

    # Check for numerical patterns that might affect text encoding

    # Number of constituents with "round" pT values (multiples of 5, 10, etc.)
    pt_round5 = np.isclose(pt_masked % 5, 0, atol=0.1) & has_particle
    properties["frac_round5_pt"] = np.where(
        n_constituents > 0,
        pt_round5.sum(axis=1) / n_constituents,
        0
    )

    # Sum of all pT values (might have rounding artifacts)
    properties["pt_sum_mod10"] = jet_pt % 10

    # Leading particle pT magnitude (order of magnitude)
    leading_pt = pt_masked.max(axis=1)
    properties["leading_pt_log"] = np.log10(leading_pt + 1)

    print(f"✓ Computed {len(properties)} advanced properties for {len(X)} jets")

    return properties


def compute_masked_skewness(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute skewness ignoring masked values."""
    n = mask.sum(axis=1)
    mean = np.where(n > 0, (data * mask).sum(axis=1) / n, 0)

    centered = (data - mean[:, None]) * mask
    m3 = (centered**3).sum(axis=1) / np.maximum(n, 1)
    std = np.sqrt((centered**2).sum(axis=1) / np.maximum(n, 1))

    skew = np.where(std > 0, m3 / (std**3), 0)
    return skew


def compute_masked_kurtosis(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute kurtosis ignoring masked values."""
    n = mask.sum(axis=1)
    mean = np.where(n > 0, (data * mask).sum(axis=1) / n, 0)

    centered = (data - mean[:, None]) * mask
    m4 = (centered**4).sum(axis=1) / np.maximum(n, 1)
    std = np.sqrt((centered**2).sum(axis=1) / np.maximum(n, 1))

    kurt = np.where(std > 0, m4 / (std**4) - 3, 0)  # Excess kurtosis
    return kurt


def compute_masked_correlation(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute correlation between x and y ignoring masked values."""
    n = mask.sum(axis=1)

    mean_x = (x * mask).sum(axis=1) / np.maximum(n, 1)
    mean_y = (y * mask).sum(axis=1) / np.maximum(n, 1)

    centered_x = (x - mean_x[:, None]) * mask
    centered_y = (y - mean_y[:, None]) * mask

    cov = (centered_x * centered_y).sum(axis=1) / np.maximum(n, 1)
    std_x = np.sqrt((centered_x**2).sum(axis=1) / np.maximum(n, 1))
    std_y = np.sqrt((centered_y**2).sum(axis=1) / np.maximum(n, 1))

    corr = np.where((std_x > 0) & (std_y > 0), cov / (std_x * std_y), 0)
    return corr


def compare_properties(
    easy_props: dict[str, np.ndarray],
    hard_props: dict[str, np.ndarray],
    output_dir: Path
) -> dict[str, dict]:
    """Compare advanced properties."""
    print("\nComparing advanced properties...")

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_results = {}

    # Statistical tests
    for prop_name, easy_vals in easy_props.items():
        hard_vals = hard_props[prop_name]

        # Remove any NaN or inf
        easy_valid = easy_vals[np.isfinite(easy_vals)]
        hard_valid = hard_vals[np.isfinite(hard_vals)]

        if len(easy_valid) < 10 or len(hard_valid) < 10:
            continue

        statistic, pvalue = stats.mannwhitneyu(easy_valid, hard_valid, alternative='two-sided')

        mean_easy, std_easy = np.mean(easy_valid), np.std(easy_valid)
        mean_hard, std_hard = np.mean(hard_valid), np.std(hard_valid)
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

    # Sort by significance
    sorted_results = sorted(
        stats_results.items(),
        key=lambda x: (x[1]["significant"], abs(x[1]["cohens_d"])),
        reverse=True
    )

    # Print summary
    print("\nAdvanced Property Comparison (sorted by significance):")
    print("="*100)
    print(f"{'Property':<30} {'Easy Mean':<12} {'Hard Mean':<12} {'Cohen d':<10} {'p-value':<12} {'Sig'}")
    print("-"*100)

    for prop_name, result in sorted_results:
        sig_marker = '***' if result['p_value'] < 0.001 else (
            '**' if result['p_value'] < 0.01 else (
            '*' if result['p_value'] < 0.05 else 'ns'))

        print(f"{prop_name:<30} "
              f"{result['easy_mean']:<12.4f} "
              f"{result['hard_mean']:<12.4f} "
              f"{result['cohens_d']:<10.2f} "
              f"{result['p_value']:<12.3e} "
              f"{sig_marker}")

    print("="*100)

    # Create plots for most significant
    significant_props = [name for name, res in sorted_results if res["significant"]][:9]

    if significant_props:
        n_plots = min(9, len(significant_props))
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for idx, prop_name in enumerate(significant_props[:9]):
            easy_vals = easy_props[prop_name]
            hard_vals = hard_props[prop_name]
            result = stats_results[prop_name]

            ax = axes[idx]
            bins = np.linspace(
                min(np.percentile(easy_vals[np.isfinite(easy_vals)], 1),
                    np.percentile(hard_vals[np.isfinite(hard_vals)], 1)),
                max(np.percentile(easy_vals[np.isfinite(easy_vals)], 99),
                    np.percentile(hard_vals[np.isfinite(hard_vals)], 99)),
                50
            )

            ax.hist(easy_vals[np.isfinite(easy_vals)], bins=bins, alpha=0.5,
                   label='Easy', density=True, color='blue')
            ax.hist(hard_vals[np.isfinite(hard_vals)], bins=bins, alpha=0.5,
                   label='Hard', density=True, color='red')
            ax.set_xlabel(prop_name.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()

            sig_marker = '***' if result['p_value'] < 0.001 else (
                '**' if result['p_value'] < 0.01 else '*')
            ax.text(0.95, 0.95, f"p={result['p_value']:.3e}\nd={result['cohens_d']:.2f} {sig_marker}",
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / "advanced_properties_significant.png", dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved significant property plots to {output_dir}")

    return dict(sorted_results)


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(
        description="Advanced property analysis for easy vs hard jets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "classification_file",
        type=Path,
        help="Path to easy/hard classification JSON file",
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
        help="Output directory",
    )

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = project_root / "data" / "qg_jets.npz"
    if args.output_dir is None:
        args.output_dir = project_root / "results" / "advanced_properties_analysis"

    print("="*80)
    print("ADVANCED PROPERTY ANALYSIS")
    print("="*80)

    # Load classification
    with open(args.classification_file) as f:
        classification = json.load(f)

    easy_indices = np.array(classification["easy_indices"], dtype=int)
    hard_indices = np.array(classification["hard_indices"], dtype=int)

    print(f"Easy jets: {len(easy_indices)}")
    print(f"Hard jets: {len(hard_indices)}")

    # Load data
    data = np.load(args.data_path)
    X_easy = data["X"][easy_indices]
    X_hard = data["X"][hard_indices]

    # Compute properties
    easy_props = compute_advanced_properties(X_easy)
    hard_props = compute_advanced_properties(X_hard)

    # Compare
    stats_results = compare_properties(easy_props, hard_props, args.output_dir)

    # Save
    output_file = args.output_dir / "advanced_properties_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats_results, f, indent=2)

    print(f"\n✓ Saved statistics to {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
