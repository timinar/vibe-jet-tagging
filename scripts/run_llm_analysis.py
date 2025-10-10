#!/usr/bin/env python3
"""
Complete LLM analysis pipeline: classify → bootstrap → plot.

This script runs the complete analysis pipeline while maintaining compatibility
with individual scripts. All intermediate files are saved and can be used
independently.

Example usage:
    # Basic run
    python scripts/run_llm_analysis.py --num_jets 100

    # Full analysis
    python scripts/run_llm_analysis.py \
      --num_jets 1000 \
      --reasoning_efforts low medium high \
      --templates simple_list with_summary_stats with_optimal_cut with_engineered_features \
      --max_concurrent 30

    # For screen/tmux
    screen -S llm_pipeline
    python scripts/run_llm_analysis.py --num_jets 1000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {description} failed: {e}")
        return False


def main():
    """Run complete LLM analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete LLM analysis pipeline: classify → bootstrap → plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Analysis parameters
    parser.add_argument(
        "--num_jets",
        type=int,
        default=100,
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
        "--max_concurrent",
        type=int,
        default=50,
        help="Maximum number of concurrent requests",
    )

    # Bootstrap parameters
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples",
    )

    # Server parameters
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
        help="API key",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model identifier",
    )

    # Path parameters
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to data file",
    )

    parser.add_argument(
        "--templates_dir",
        type=Path,
        default=None,
        help="Path to templates directory",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )

    # Pipeline control
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Skip LLM analysis (use existing results)",
    )

    parser.add_argument(
        "--skip_bootstrap",
        action="store_true",
        help="Skip bootstrap (use existing bootstrap results)",
    )

    parser.add_argument(
        "--skip_plot",
        action="store_true",
        help="Skip plotting",
    )

    parser.add_argument(
        "--analysis_output",
        type=Path,
        default=None,
        help="Existing analysis JSON file (if --skip_analysis)",
    )

    args = parser.parse_args()

    # Set default paths
    if args.data_path is None:
        args.data_path = project_root / "data" / "qg_jets.npz"
    if args.templates_dir is None:
        args.templates_dir = project_root / "templates"
    if args.output_dir is None:
        args.output_dir = project_root / "results"
        args.output_dir.mkdir(exist_ok=True)

    # Generate timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_file = args.output_dir / f"llm_analysis_{timestamp}.json"
    bootstrap_file = args.output_dir / f"llm_analysis_{timestamp}_bootstrap.json"
    plot_file = args.output_dir / f"llm_analysis_{timestamp}_bootstrap.png"

    print("="*80)
    print("LLM ANALYSIS PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of jets:      {args.num_jets}")
    print(f"  Templates:           {', '.join(args.templates)}")
    print(f"  Reasoning efforts:   {', '.join(args.reasoning_efforts)}")
    print(f"  Max concurrent:      {args.max_concurrent}")
    print(f"  Bootstrap samples:   {args.n_bootstrap}")
    print(f"  Model:               {args.model_name}")
    print(f"\nOutput files:")
    print(f"  Analysis:            {analysis_file}")
    print(f"  Bootstrap:           {bootstrap_file}")
    print(f"  Plot:                {plot_file}")
    print("="*80)

    start_time = time.time()

    # Step 1: LLM Analysis
    if not args.skip_analysis:
        cmd = [
            "uv", "run", "python", str(project_root / "scripts" / "analyze_llm_templates.py"),
            "--num_jets", str(args.num_jets),
            "--reasoning_efforts", *args.reasoning_efforts,
            "--templates", *args.templates,
            "--max_concurrent", str(args.max_concurrent),
            "--base_url", args.base_url,
            "--api_key", args.api_key,
            "--model_name", args.model_name,
            "--data_path", str(args.data_path),
            "--templates_dir", str(args.templates_dir),
            "--output", str(analysis_file),
        ]

        if not run_command(cmd, "LLM Classification"):
            print("\n❌ Pipeline failed at LLM classification step")
            sys.exit(1)
    else:
        if args.analysis_output and args.analysis_output.exists():
            analysis_file = args.analysis_output
            print(f"\n✓ Using existing analysis: {analysis_file}")
        else:
            print(f"\n❌ No analysis file specified or file not found")
            sys.exit(1)

    # Step 2: Bootstrap
    if not args.skip_bootstrap:
        cmd = [
            "uv", "run", "python", str(project_root / "scripts" / "bootstrap_llm_results.py"),
            str(analysis_file),
            "--n_bootstrap", str(args.n_bootstrap),
        ]

        if not run_command(cmd, "Bootstrap Error Bars"):
            print("\n❌ Pipeline failed at bootstrap step")
            sys.exit(1)
    else:
        print(f"\n✓ Skipping bootstrap (using existing if available)")

    # Step 3: Plot
    if not args.skip_plot:
        cmd = [
            "uv", "run", "python", str(project_root / "scripts" / "plot_llm_analysis.py"),
            str(bootstrap_file),
            "--output", str(plot_file),
        ]

        if not run_command(cmd, "Generate Plots"):
            print("\n⚠️  Plotting failed, but analysis and bootstrap completed")
    else:
        print(f"\n✓ Skipping plot generation")

    total_time = time.time() - start_time

    # Final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nGenerated files:")
    if analysis_file.exists():
        print(f"  ✓ Analysis:   {analysis_file}")
    if bootstrap_file.exists():
        print(f"  ✓ Bootstrap:  {bootstrap_file}")
    if plot_file.exists():
        print(f"  ✓ Plot:       {plot_file}")
    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"  - View plot: open {plot_file}")
    print(f"  - Replot:    python scripts/plot_llm_analysis.py {bootstrap_file}")
    print(f"  - Compare:   python scripts/bootstrap_error_bars.py")
    print("="*80)


if __name__ == "__main__":
    main()
