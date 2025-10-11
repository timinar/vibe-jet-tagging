#!/bin/bash
# Complete pipeline for easy vs hard jets investigation

set -e  # Exit on error

echo "========================================================================"
echo "EASY vs HARD JETS INVESTIGATION PIPELINE"
echo "========================================================================"

# Configuration
NUM_JETS=1000
TEMPLATE="with_optimal_cut"
REASONING="low"
MAX_CONCURRENT=50

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Configuration:"
echo "  - Number of jets: $NUM_JETS"
echo "  - Template: $TEMPLATE"
echo "  - Reasoning: $REASONING"
echo "  - Max concurrent: $MAX_CONCURRENT"
echo ""

# Step 1: Run LLM analysis with jet tracking
echo "========================================================================"
echo "STEP 1: Running LLM Analysis (with jet index tracking)"
echo "========================================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ANALYSIS_FILE="results/llm_analysis_${TIMESTAMP}_tracked.json"

uv run python scripts/analyze_llm_templates.py \
  --num_jets $NUM_JETS \
  --templates $TEMPLATE \
  --reasoning_efforts $REASONING \
  --max_concurrent $MAX_CONCURRENT \
  --output "$ANALYSIS_FILE"

echo ""
echo "✓ Analysis complete: $ANALYSIS_FILE"

# Step 2: Run bootstrap
echo ""
echo "========================================================================"
echo "STEP 2: Computing Bootstrap Confidence Intervals"
echo "========================================================================"

uv run python scripts/bootstrap_llm_results.py "$ANALYSIS_FILE" --n_bootstrap 1000

BOOTSTRAP_FILE="${ANALYSIS_FILE%.json}_bootstrap.json"
echo "✓ Bootstrap complete: $BOOTSTRAP_FILE"

# Step 3: Classify easy vs hard jets
echo ""
echo "========================================================================"
echo "STEP 3: Classifying Easy vs Hard Jets"
echo "========================================================================"

uv run python scripts/analyze_easy_hard_jets.py \
  "$ANALYSIS_FILE" \
  --template $TEMPLATE \
  --reasoning_effort $REASONING

CLASSIFICATION_FILE="results/easy_hard_analysis/easy_hard_classification.json"
echo "✓ Classification complete: $CLASSIFICATION_FILE"

# Step 4: Validation experiments
echo ""
echo "========================================================================"
echo "STEP 4: Running Validation Experiments"
echo "========================================================================"
echo "This will run LLM classification on:"
echo "  - 100 easy jets only"
echo "  - 100 hard jets only"
echo "  - 50 easy + 50 hard jets (mixed)"
echo ""
read -p "Continue with validation experiments? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]
then
    uv run python scripts/validate_easy_hard_hypothesis.py \
      "$CLASSIFICATION_FILE" \
      --experiments easy_only hard_only mixed \
      --n_jets 100 \
      --max_concurrent $MAX_CONCURRENT

    echo "✓ Validation experiments complete"
else
    echo "Skipping validation experiments"
fi

# Summary
echo ""
echo "========================================================================"
echo "INVESTIGATION COMPLETE"
echo "========================================================================"
echo "Generated files:"
echo "  - Analysis: $ANALYSIS_FILE"
echo "  - Bootstrap: $BOOTSTRAP_FILE"
echo "  - Classification: $CLASSIFICATION_FILE"
echo "  - Property plots: results/easy_hard_analysis/easy_vs_hard_properties.png"
echo "  - Validation: results/validation_experiments/"
echo ""
echo "Next steps:"
echo "  1. Review property analysis plots"
echo "  2. Check validation results"
echo "  3. Design template improvements based on findings"
echo "========================================================================"
