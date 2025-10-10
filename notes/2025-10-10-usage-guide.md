# Usage Guide: Complete LLM Analysis Pipeline

**Date**: 2025-10-10
**Branch**: `experiment/gpt-oss`

## Quick Start: One-Command Pipeline

The easiest way to run a complete analysis:

```bash
# Start vLLM server first
screen -S vllm_server
vllm serve openai/gpt-oss-120b
# Ctrl+A, D to detach

# Run complete pipeline
screen -S llm_pipeline
uv run python scripts/run_llm_analysis.py --num_jets 1000 --max_concurrent 30
# Ctrl+A, D to detach
```

This will:
1. Run LLM classification on 1000 jets
2. Bootstrap error bars (1000 samples)
3. Generate plots with error bars
4. Save all intermediate files

**Output files:**
- `results/llm_analysis_TIMESTAMP.json` - Raw analysis results
- `results/llm_analysis_TIMESTAMP_bootstrap.json` - With error bars
- `results/llm_analysis_TIMESTAMP_bootstrap.png` - Plots

## Advanced Usage

### Full Configuration

```bash
uv run python scripts/run_llm_analysis.py \
  --num_jets 1000 \
  --reasoning_efforts low medium high \
  --templates simple_list with_summary_stats with_optimal_cut with_engineered_features \
  --max_concurrent 30 \
  --n_bootstrap 1000
```

### Skip Steps (Reuse Existing Results)

```bash
# Skip analysis, use existing results
uv run python scripts/run_llm_analysis.py \
  --skip_analysis \
  --analysis_output results/llm_analysis_20251010_220015.json

# Skip bootstrap (if already done)
uv run python scripts/run_llm_analysis.py \
  --skip_bootstrap \
  --analysis_output results/llm_analysis_20251010_220015.json

# Only plot (no analysis or bootstrap)
uv run python scripts/run_llm_analysis.py \
  --skip_analysis \
  --skip_bootstrap \
  --analysis_output results/llm_analysis_20251010_220015.json
```

## Individual Scripts (Maximum Flexibility)

You can also run each step independently:

### 1. LLM Analysis

```bash
uv run python scripts/analyze_llm_templates.py \
  --num_jets 1000 \
  --reasoning_efforts low medium \
  --templates simple_list with_summary_stats \
  --max_concurrent 30
```

**Saved to:** `results/llm_analysis_TIMESTAMP.json`

### 2. Bootstrap

```bash
uv run python scripts/bootstrap_llm_results.py \
  results/llm_analysis_*.json \
  --n_bootstrap 1000
```

**Saved to:** `results/llm_analysis_*_bootstrap.json`

### 3. Plot

```bash
# With bootstrap
uv run python scripts/plot_llm_analysis.py \
  results/llm_analysis_*_bootstrap.json

# Without bootstrap (works too!)
uv run python scripts/plot_llm_analysis.py \
  results/llm_analysis_*.json
```

**Saved to:** Same name as input but `.png`

### 4. Baseline Bootstrap

```bash
uv run python scripts/bootstrap_error_bars.py
```

**Saved to:** `results/bootstrap_baselines.npz`

## Plot Features

The generated plots show:

1. **Accuracy by Template & Effort** (top-left)
   - Grouped bar chart with error bars
   - Random baseline at 0.5

2. **AUC by Template & Effort** (top-right)
   - Grouped bar chart with error bars
   - Random baseline at 0.5

3. **Generation Time** (middle-left)
   - Time cost for each configuration
   - Shows computational trade-offs

4. **Reasoning Tokens** (middle-right)
   - Token usage by configuration
   - Estimated from response structure

5. **Performance vs Time Trade-off** (wide bottom panel)
   - **Lines connect same template across reasoning efforts**
   - Shows clear trade-off: more time → better performance
   - Annotated with effort labels (L/M/H)
   - Error bars when bootstrap data available
   - Easy to identify optimal configurations

## Concurrency Settings

Choose based on your setup:

| Setup | `--max_concurrent` | Notes |
|-------|-------------------|-------|
| Single GPU | 20-30 | Conservative |
| Multi-GPU | 50 | Default |
| Shared HPC | 10-20 | Be considerate |
| Dedicated server | 50-100 | Max performance |

## Monitoring Progress

```bash
# Check on running pipeline
screen -r llm_pipeline

# Check server
screen -r vllm_server

# GPU usage
watch -n 1 nvidia-smi

# Disk space (results can be large)
du -sh results/
```

## Expected Run Times (1000 jets)

Approximate times for 1000 jets with max_concurrent=30:

| Configuration | Time per jet | Total time |
|--------------|--------------|------------|
| simple_list (low) | ~0.15s | ~2.5 min |
| simple_list (medium) | ~0.75s | ~12 min |
| with_summary_stats (low) | ~0.15s | ~2.5 min |
| with_summary_stats (medium) | ~0.75s | ~12 min |

**Full 8 configs (4 templates × 2 efforts):** ~60-90 minutes

**Bootstrap (1000 samples):** ~1-2 minutes per config

## Troubleshooting

### Connection Errors
- Reduce `--max_concurrent` (try 20 or 10)
- Check server is running: `screen -r vllm_server`

### Out of Memory
- Check GPU usage: `nvidia-smi`
- Reduce concurrent requests or restart server

### Slow Performance
- Check GPU utilization (should be high)
- Increase `--max_concurrent` if GPU underutilized
- Check network/disk I/O

### Plot Issues
- Ensure matplotlib is installed: `uv pip install matplotlib`
- For headless servers, plots still save to file

## File Compatibility

All intermediate files are compatible:
- Analyze → Bootstrap → Plot (full pipeline)
- Analyze → Plot (skip bootstrap)
- Bootstrap → Plot (rerun bootstrap with different samples)
- Plot only (regenerate plots from existing data)

This design allows maximum flexibility and efficiency!
