# Usage Guide: LLM Analysis Pipeline

**Date**: 2025-10-10
**Branch**: `experiment/gpt-oss`

## Quick Start

```bash
# 1. Start vLLM server
screen -S vllm_server
vllm serve openai/gpt-oss-120b
# Ctrl+A, D to detach

# 2. Run complete pipeline
screen -S llm_pipeline
uv run python scripts/run_llm_analysis.py --num_jets 300 --templates simple_list with_summary_stats with_optimal_cut --max_concurrent 100
# Ctrl+A, D to detach
```

**Output files:**
- `results/llm_analysis_TIMESTAMP.json` - Raw results
- `results/llm_analysis_TIMESTAMP_bootstrap.json` - With error bars
- `results/llm_analysis_TIMESTAMP_bootstrap.png` - Plots

## Full Configuration

```bash
uv run python scripts/run_llm_analysis.py \
  --num_jets 1000 \
  --reasoning_efforts low medium high \
  --templates simple_list with_summary_stats with_optimal_cut with_engineered_features \
  --max_concurrent 30 \
  --n_bootstrap 1000
```

## Individual Scripts

```bash
# 1. Analysis only
uv run python scripts/analyze_llm_templates.py --num_jets 1000

# 2. Bootstrap existing results
uv run python scripts/bootstrap_llm_results.py results/llm_analysis_*.json

# 3. Plot (works with or without bootstrap)
uv run python scripts/plot_llm_analysis.py results/llm_analysis_*.json

# 4. Baseline comparison
uv run python scripts/bootstrap_error_bars.py
```

## Concurrency Settings

| Setup | `--max_concurrent` |
|-------|-------------------|
| Single GPU | 20-30 |
| Multi-GPU | 50 (default) |
| Shared HPC | 10-20 |

## Expected Times (1000 jets, max_concurrent=30)

- **Low effort:** ~2-3 min per template
- **Medium effort:** ~10-15 min per template
- **Full 8 configs:** ~60-90 min
- **Bootstrap:** ~1-2 min per config

## Monitoring

```bash
screen -r llm_pipeline  # Check progress
screen -r vllm_server   # Check server
nvidia-smi              # GPU usage
```

## Troubleshooting

**Connection errors:** Reduce `--max_concurrent` to 20 or 10
**Out of memory:** Check `nvidia-smi`, reduce concurrent requests
**Slow:** Increase `--max_concurrent` if GPU underutilized

## Recent Fixes

**Event loop cleanup (2025-10-10):** Fixed "Event loop is closed" errors that occurred when running multiple sequential configurations. The classifier now uses a persistent event loop that stays alive across predict() calls, preventing cleanup issues with high concurrency.
