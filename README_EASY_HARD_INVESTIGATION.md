# Easy vs Hard Jets Investigation

## Quick Start

### Phase 1: Analysis with Tracking (Currently Running)
```bash
# This is already running in the background
# Check progress:
ls -lh results/llm_analysis_reference_tracked.json

# Expected time: ~2-3 minutes for 1000 jets
```

### Phase 2: Property Analysis (Run After Phase 1)
```bash
uv run python scripts/analyze_easy_hard_jets.py \
  results/llm_analysis_reference_tracked.json \
  --template with_optimal_cut \
  --reasoning_effort low
```

**Output**:
- `results/easy_hard_analysis/easy_hard_classification.json`
- `results/easy_hard_analysis/easy_vs_hard_properties.png`
- Console output with statistical comparison

### Phase 3: Validation Experiments (Run After Phase 2)
```bash
uv run python scripts/validate_easy_hard_hypothesis.py \
  results/easy_hard_analysis/easy_hard_classification.json \
  --experiments easy_only hard_only mixed \
  --n_jets 100
```

**Output**:
- `results/validation_experiments/validation_easy_only.json`
- `results/validation_experiments/validation_hard_only.json`
- `results/validation_experiments/validation_mixed.json`
- `results/validation_experiments/validation_summary.json`

### Or: Run Complete Pipeline
```bash
./scripts/run_easy_hard_investigation.sh
```

## Understanding the Results

### Property Analysis

The script will show you which properties differ significantly between easy and hard jets. Look for:

1. **Large Cohen's d** (|d| > 0.5): Substantial effect size
2. **Small p-value** (p < 0.001): Highly significant difference
3. **Distribution separation**: Visual inspection of histograms

**Example interpretation**:
- If hard jets have **higher multiplicity** → LLM struggles with complex jets
- If hard jets have **wider angular distribution** → LLM needs better spatial reasoning
- If hard jets have **lower pT** → LLM performance degrades at lower energy scales

### Validation Results

Compare AUC across experiments:

| Experiment | Expected AUC | Interpretation |
|------------|--------------|----------------|
| Easy only  | 0.85-0.90    | Confirms these jets are genuinely easy |
| Hard only  | 0.55-0.65    | Near random → these are truly challenging |
| Mixed      | ~0.70-0.75   | Intermediate, matches 1000-jet baseline |

**If hypothesis is confirmed**:
- Small samples (100) get "lucky" with easy jets
- Large samples (1000+) show true performance
- Error bars should be adjusted for sample composition bias

## What Each Script Does

### 1. analyze_llm_templates.py (Modified)
- **Input**: Jet dataset, LLM API
- **Output**: Predictions + jet indices
- **New features**: `--random_sample`, `--seed` flags

### 2. analyze_easy_hard_jets.py
- **Input**: Analysis results with jet indices
- **Output**: Easy/hard classification + property comparison
- **Key functions**:
  - `classify_jets_easy_hard()`: Correct vs incorrect
  - `compute_jet_properties()`: 8 kinematic/substructure variables
  - `compare_properties()`: Statistical tests + plots

### 3. validate_easy_hard_hypothesis.py
- **Input**: Easy/hard classification
- **Output**: Targeted LLM experiments
- **Experiments**: Easy-only, hard-only, mixed samples

## Next Steps After Validation

### If Hypothesis is Confirmed

1. **Characterize hard jets**: Document their properties
2. **Design targeted improvements**:
   - Add guidance for high-multiplicity jets
   - Improve spatial reasoning for wide jets
   - Add examples of challenging cases

3. **Create enhanced template**: `templates/with_optimal_cut_enhanced.md`

4. **Test improvement**:
   ```bash
   # Re-run on hard jets with new template
   python scripts/analyze_llm_templates.py \
     --num_jets 100 \
     --templates with_optimal_cut_enhanced \
     --output results/enhanced_template_test.json
   ```

5. **Measure improvement**:
   - Target: +5-10% AUC on hard jets
   - Reduced variance across sample sizes

### If Hypothesis is Not Confirmed

Investigate alternative explanations:
- Random sampling artifacts
- Model temperature/randomness
- Batch effects in LLM inference
- Data preprocessing issues

## Troubleshooting

### Analysis Taking Too Long
```bash
# Check if still running
ps aux | grep analyze_llm_templates

# Check server
curl http://localhost:8000/v1/models

# Monitor GPU usage
nvidia-smi
```

### Out of Memory
- Reduce `--max_concurrent` (default: 50)
- Check GPU memory: `nvidia-smi`

### Server Connection Issues
```bash
# Restart server
vllm serve openai/gpt-oss-120b --port 8000
```

### Missing Dependencies
```bash
uv sync
```

## Files and Directories

```
scripts/
  ├── analyze_llm_templates.py          # Modified for tracking
  ├── analyze_easy_hard_jets.py         # New: Property analysis
  ├── validate_easy_hard_hypothesis.py  # New: Validation experiments
  └── run_easy_hard_investigation.sh    # New: Complete pipeline

results/
  ├── llm_analysis_reference_tracked.json           # Reference run
  ├── easy_hard_analysis/
  │   ├── easy_hard_classification.json             # Jet classification
  │   └── easy_vs_hard_properties.png               # Property plots
  └── validation_experiments/
      ├── validation_easy_only.json                 # Easy jets test
      ├── validation_hard_only.json                 # Hard jets test
      ├── validation_mixed.json                     # Mixed test
      └── validation_summary.json                   # Summary

notes/
  ├── 2025-10-11-easy-hard-jets-investigation-plan.md  # Detailed plan
  └── 2025-10-11-implementation-summary.md             # Implementation notes
```

## Monitoring Progress

### During Analysis
```bash
# Check if complete
ls -lh results/llm_analysis_reference_tracked.json

# Monitor in real-time
watch -n 5 'ls -lh results/*.json | tail -3'
```

### During Validation
```bash
# Check validation experiments
ls -lh results/validation_experiments/

# Summary
cat results/validation_experiments/validation_summary.json
```

## Timeline

- **Phase 1** (Running): 2-3 minutes
- **Phase 2**: < 1 minute
- **Phase 3**: 5-8 minutes (3 LLM experiments)
- **Total**: ~10-15 minutes

## Questions?

Check the detailed documentation:
- `notes/2025-10-11-easy-hard-jets-investigation-plan.md` - Full plan
- `notes/2025-10-11-implementation-summary.md` - Technical details
- `notes/2025-10-10-usage-guide.md` - General usage guide
