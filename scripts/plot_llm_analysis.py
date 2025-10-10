#!/usr/bin/env python3
"""
Plot results from LLM classifier analysis.

This script creates comprehensive visualizations comparing different templates
and reasoning efforts for the LocalLLMClassifier on jet tagging.

Example usage:
    python scripts/plot_llm_analysis.py results/llm_analysis_20250110_123456.json

    # Save to specific location
    python scripts/plot_llm_analysis.py results/llm_analysis_20250110_123456.json --output plots/analysis.png

    # Show interactive plot
    python scripts/plot_llm_analysis.py results/llm_analysis_20250110_123456.json --show
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10


def load_results(results_path: Path) -> dict:
    """Load analysis results from JSON file.

    Parameters
    ----------
    results_path : Path
        Path to results JSON file

    Returns
    -------
    data : dict
        Results data including metadata and metrics
    """
    with open(results_path) as f:
        data = json.load(f)

    print(f"✓ Loaded results from {results_path}")
    print(f"  Metadata:")
    for key, value in data["metadata"].items():
        if key not in ["data_path", "templates_dir"]:
            print(f"    {key}: {value}")
    print(f"  Total configurations: {len(data['results'])}")

    return data


def plot_comprehensive_comparison(data: dict, output_path: Path | None = None, show: bool = False):
    """Create comprehensive comparison plots.

    Parameters
    ----------
    data : dict
        Results data
    output_path : Path, optional
        Path to save the figure
    show : bool
        Whether to show the interactive plot
    """
    results = data["results"]
    metadata = data["metadata"]

    # Extract unique templates and efforts
    templates = sorted(set(r["template"] for r in results))
    efforts = sorted(set(r["reasoning_effort"] for r in results))

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Color schemes
    effort_colors = {
        "low": "steelblue",
        "medium": "coral",
        "high": "darkred",
    }

    # Prepare data for grouped bar charts
    x_pos = np.arange(len(templates))
    width = 0.8 / len(efforts)

    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    for i, effort in enumerate(efforts):
        accuracies = []
        for template in templates:
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            accuracies.append(matching[0]["accuracy"] if matching else 0)

        offset = width * (i - len(efforts) / 2 + 0.5)
        bars = ax1.bar(
            x_pos + offset,
            accuracies,
            width,
            label=f"effort={effort}",
            color=effort_colors.get(effort, "gray"),
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Accuracy", fontweight="bold")
    ax1.set_title("Accuracy by Template and Reasoning Effort", fontweight="bold", fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(templates, rotation=15, ha="right")
    ax1.legend()
    ax1.set_ylim([0, 1.0])
    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random")

    # 2. AUC Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    for i, effort in enumerate(efforts):
        aucs = []
        for template in templates:
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            aucs.append(matching[0]["auc"] if matching else 0)

        offset = width * (i - len(efforts) / 2 + 0.5)
        bars = ax2.bar(
            x_pos + offset,
            aucs,
            width,
            label=f"effort={effort}",
            color=effort_colors.get(effort, "gray"),
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.set_ylabel("AUC", fontweight="bold")
    ax2.set_title("AUC by Template and Reasoning Effort", fontweight="bold", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(templates, rotation=15, ha="right")
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.3, label="Random")

    # 3. Generation Time Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    for i, effort in enumerate(efforts):
        times = []
        for template in templates:
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            times.append(matching[0]["total_generation_time"] if matching else 0)

        offset = width * (i - len(efforts) / 2 + 0.5)
        bars = ax3.bar(
            x_pos + offset,
            times,
            width,
            label=f"effort={effort}",
            color=effort_colors.get(effort, "gray"),
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax3.set_ylabel("Generation Time (seconds)", fontweight="bold")
    ax3.set_title(
        f"Generation Time by Template and Reasoning Effort ({metadata['num_jets']} jets)",
        fontweight="bold",
        fontsize=12,
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(templates, rotation=15, ha="right")
    ax3.legend()

    # 4. Reasoning Tokens Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    for i, effort in enumerate(efforts):
        reasoning_tokens_list = []
        for template in templates:
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            reasoning_tokens_list.append(matching[0]["reasoning_tokens"] if matching else 0)

        offset = width * (i - len(efforts) / 2 + 0.5)
        bars = ax4.bar(
            x_pos + offset,
            reasoning_tokens_list,
            width,
            label=f"effort={effort}",
            color=effort_colors.get(effort, "gray"),
            alpha=0.8,
            edgecolor="black",
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax4.set_ylabel("Reasoning Tokens (estimate)", fontweight="bold")
    ax4.set_title("Reasoning Token Usage by Template and Effort", fontweight="bold", fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(templates, rotation=15, ha="right")
    ax4.legend()

    # 5. Performance vs Time Trade-off
    ax5 = fig.add_subplot(gs[2, 0])
    for template in templates:
        for effort in efforts:
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            if matching:
                result = matching[0]
                marker = "o" if effort == "low" else ("s" if effort == "medium" else "^")
                ax5.scatter(
                    result["total_generation_time"],
                    result["auc"],
                    s=200,
                    marker=marker,
                    color=effort_colors.get(effort, "gray"),
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=2,
                    label=f"{template} ({effort})",
                )

    ax5.set_xlabel("Generation Time (seconds)", fontweight="bold")
    ax5.set_ylabel("AUC", fontweight="bold")
    ax5.set_title(
        "Performance vs Time Trade-off (○=low, □=medium, △=high)",
        fontweight="bold",
        fontsize=12,
    )
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Heatmap of AUC scores
    ax6 = fig.add_subplot(gs[2, 1])

    # Create AUC matrix
    auc_matrix = np.zeros((len(templates), len(efforts)))
    for i, template in enumerate(templates):
        for j, effort in enumerate(efforts):
            matching = [r for r in results if r["template"] == template and r["reasoning_effort"] == effort]
            if matching:
                auc_matrix[i, j] = matching[0]["auc"]

    # Plot heatmap
    im = ax6.imshow(auc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

    # Set ticks
    ax6.set_xticks(np.arange(len(efforts)))
    ax6.set_yticks(np.arange(len(templates)))
    ax6.set_xticklabels(efforts)
    ax6.set_yticklabels(templates)

    # Add text annotations
    for i in range(len(templates)):
        for j in range(len(efforts)):
            text = ax6.text(
                j,
                i,
                f"{auc_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=9,
            )

    ax6.set_title("AUC Heatmap: Template × Reasoning Effort", fontweight="bold", fontsize=12)
    ax6.set_xlabel("Reasoning Effort", fontweight="bold")
    ax6.set_ylabel("Template", fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax6)
    cbar.set_label("AUC", fontweight="bold")

    # Overall title
    fig.suptitle(
        f"LLM Classifier Analysis: {metadata['model_name']}\n"
        f"{metadata['num_jets']} jets | {len(results)} configurations",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n✓ Figure saved to {output_path}")

    if show:
        plt.show()

    plt.close()


def print_summary_table(data: dict):
    """Print summary table of results.

    Parameters
    ----------
    data : dict
        Results data
    """
    results = data["results"]

    print(f"\n{'='*100}")
    print("DETAILED RESULTS")
    print(f"{'='*100}")
    print(
        f"{'Template':<25} {'Effort':<10} {'Accuracy':<10} {'AUC':<10} "
        f"{'Time (s)':<12} {'Reasoning Tokens':<15}"
    )
    print("-" * 100)

    # Sort by AUC descending
    sorted_results = sorted(results, key=lambda x: x["auc"], reverse=True)

    for result in sorted_results:
        print(
            f"{result['template']:<25} "
            f"{result['reasoning_effort']:<10} "
            f"{result['accuracy']:<10.4f} "
            f"{result['auc']:<10.4f} "
            f"{result['total_generation_time']:<12.2f} "
            f"{result['reasoning_tokens']:<15,}"
        )

    print("=" * 100)

    # Print insights
    best = sorted_results[0]
    print(f"\n🏆 Best Configuration:")
    print(f"   Template: {best['template']}")
    print(f"   Reasoning effort: {best['reasoning_effort']}")
    print(f"   AUC: {best['auc']:.4f}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    print(f"   Generation time: {best['total_generation_time']:.2f}s")

    # Find fastest
    fastest = min(results, key=lambda x: x["total_generation_time"])
    print(f"\n⚡ Fastest Configuration:")
    print(f"   Template: {fastest['template']}")
    print(f"   Reasoning effort: {fastest['reasoning_effort']}")
    print(f"   AUC: {fastest['auc']:.4f}")
    print(f"   Generation time: {fastest['total_generation_time']:.2f}s")

    # Analyze reasoning effort impact
    templates = sorted(set(r["template"] for r in results))
    efforts = sorted(set(r["reasoning_effort"] for r in results))

    if len(efforts) > 1:
        print(f"\n📊 Reasoning Effort Impact:")
        for template in templates:
            template_results = [r for r in results if r["template"] == template]
            if len(template_results) >= 2:
                aucs_by_effort = {r["reasoning_effort"]: r["auc"] for r in template_results}
                times_by_effort = {r["reasoning_effort"]: r["total_generation_time"] for r in template_results}

                if "low" in aucs_by_effort and "high" in aucs_by_effort:
                    delta_auc = aucs_by_effort["high"] - aucs_by_effort["low"]
                    delta_time = times_by_effort["high"] - times_by_effort["low"]
                    print(
                        f"   {template:<25} Low→High: ΔAUC={delta_auc:+.4f}, ΔTime={delta_time:+.2f}s"
                    )


def main():
    """Main plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot LLM classifier analysis results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to results JSON file from analyze_llm_templates.py",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path (default: same directory as results, with .png extension)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot",
    )

    args = parser.parse_args()

    # Check if results file exists
    if not args.results_file.exists():
        print(f"❌ Results file not found: {args.results_file}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        args.output = args.results_file.with_suffix(".png")

    # Load results
    print(f"Loading results from {args.results_file}...")
    data = load_results(args.results_file)

    # Print summary table
    print_summary_table(data)

    # Create plots
    print(f"\nGenerating plots...")
    plot_comprehensive_comparison(data, output_path=args.output, show=args.show)

    print(f"\n✓ Plotting complete")


if __name__ == "__main__":
    main()
