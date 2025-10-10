# Notes Index

Documentation and notes for the vibe-jet-tagging project.

## Recent Notes (October 2025)

### 2025-10-10
- **[async-fixes.md](2025-10-10-async-fixes.md)** - Solutions for async processing issues (concurrency control, retry logic)
- **[running-llm-server.md](2025-10-10-running-llm-server.md)** - Guide for running vLLM server in background (screen/tmux/nohup)

## Older Notes

### Integration and Setup
- **[INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)** - Summary of local LLM integration
- **[local_llm_integration.md](local_llm_integration.md)** - Details on LocalLLMClassifier implementation
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide for the project

### Technical Details
- **[response_structure_guide.md](response_structure_guide.md)** - OpenAI Responses API structure
- **[parser_fix_analysis.md](parser_fix_analysis.md)** - Analysis of parser fixes
- **[gemini_refactor_summary.md](gemini_refactor_summary.md)** - Gemini classifier refactoring
- **[gemini_thinking_budget_fix.md](gemini_thinking_budget_fix.md)** - Gemini thinking budget fixes

### Project Documentation
- **[project_overview.md](project_overview.md)** - High-level project overview
- **[physics_and_methods.md](physics_and_methods.md)** - Physics background and methods
- **[baseline_usage.md](baseline_usage.md)** - Baseline classifier usage
- **[llm_classifier_implementation.md](llm_classifier_implementation.md)** - LLM classifier details

## Quick Links

### Running Analysis
```bash
# Start vLLM server (see running-llm-server.md)
screen -S vllm_server
vllm serve openai/gpt-oss-120b

# Run analysis (see async-fixes.md for concurrency tuning)
screen -S llm_analysis
uv run python scripts/analyze_llm_templates.py \
  --num_jets 1000 \
  --max_concurrent 30
```

### Generating Plots
```bash
uv run python scripts/plot_llm_analysis.py results/llm_analysis_*.json
```

## Note Naming Convention

All new notes should follow the format: `YYYY-MM-DD-brief-description.md`

Example: `2025-10-10-async-fixes.md`
