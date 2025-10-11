# Easy vs Hard Jets Investigation - Implementation Summary

**Date**: 2025-10-11
**Status**: Phase 1 Complete, Analysis Running

## What We've Built

### 1. Enhanced Analysis Pipeline ✅

**Modified**: `scripts/analyze_llm_templates.py`

**New Features**:
- Jet index tracking: Each analyzed jet's original dataset index is now saved
- Random sampling option: `--random_sample` flag for non-sequential sampling
- Reproducible seeding: `--seed` parameter for consistent random samples
- Enhanced metadata: Tracks sampling method in output files

**Usage**:
```bash
# Sequential sampling (default)
python scripts/analyze_llm_templates.py --num_jets 1000 --templates with_optimal_cut

# Random sampling
python scripts/analyze_llm_templates.py --num_jets 1000 --random_sample --seed 42
```

### 2. Easy/Hard Classification & Property Analysis ✅

**New Script**: `scripts/analyze_easy_hard_jets.py`

**Capabilities**:
- Loads LLM analysis results with jet indices
- Classifies jets as "easy" (correctly predicted) vs "hard" (incorrectly predicted)
- Computes comprehensive jet properties:
  - **Basic kinematics**: pT, mass, multiplicity, leading particle fraction
  - **Substructure**: angular width, girth, pT dispersion, pT^D
- Statistical comparison:
  - Mann-Whitney U tests
  - Cohen's d effect sizes
  - Significance testing
- Generates comparison plots showing property distributions

**Usage**:
```bash
python scripts/analyze_easy_hard_jets.py \
  results/llm_analysis_reference_tracked.json \
  --template with_optimal_cut \
  --reasoning_effort low
```

**Output**:
- `results/easy_hard_analysis/easy_hard_classification.json` - Jet indices and labels
- `results/easy_hard_analysis/easy_vs_hard_properties.png` - Property comparison plots
- Statistical summary printed to console

### 3. Hypothesis Validation Framework ✅

**New Script**: `scripts/validate_easy_hard_hypothesis.py`

**Experiments**:
1. **Easy Jets Only**: Test on 100 jets that were previously classified correctly
2. **Hard Jets Only**: Test on 100 jets that were previously classified incorrectly
3. **Mixed Sample**: Test on 50 easy + 50 hard jets

**Expected Results** (if hypothesis is correct):
- Easy jets → AUC ≈ 0.85-0.90 (like original 100-jet sample)
- Hard jets → AUC ≈ 0.55-0.65 (near random)
- Mixed → Intermediate AUC

**Usage**:
```bash
python scripts/validate_easy_hard_hypothesis.py \
  results/easy_hard_analysis/easy_hard_classification.json \
  --experiments easy_only hard_only mixed \
  --n_jets 100
```

**Output**:
- Creates subset data files for each experiment
- Runs LLM analysis on each subset
- Generates validation summary with AUC comparison

### 4. Complete Pipeline Script ✅

**New Script**: `scripts/run_easy_hard_investigation.sh`

**What It Does**:
- Runs complete investigation pipeline end-to-end
- Step 1: LLM analysis with tracking (1000 jets)
- Step 2: Bootstrap confidence intervals
- Step 3: Classify jets and analyze properties
- Step 4: Validation experiments (interactive)

**Usage**:
```bash
./scripts/run_easy_hard_investigation.sh
```

## Current Status

### Running Now ⏳
- **Reference analysis**: 1000 jets with `with_optimal_cut` template, low reasoning
- **Estimated time**: ~2-3 minutes (based on previous 1000-jet runs)
- **Background job ID**: 876d36

### Next Steps (Automated)
Once the reference analysis completes, you can run:

1. **Property Analysis**:
   ```bash
   uv run python scripts/analyze_easy_hard_jets.py \
     results/llm_analysis_reference_tracked.json
   ```

2. **Validation Experiments**:
   ```bash
   uv run python scripts/validate_easy_hard_hypothesis.py \
     results/easy_hard_analysis/easy_hard_classification.json
   ```

Or simply run the complete pipeline:
```bash
./scripts/run_easy_hard_investigation.sh
```

## Expected Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Implementation | 1-2 hours | ✅ Done |
| 2 | Reference analysis (1000 jets) | 2-3 min | ⏳ Running |
| 3 | Property analysis | < 1 min | ⏸️ Waiting |
| 4 | Validation experiments (300 jets total) | 5-8 min | ⏸️ Waiting |
| 5 | Template improvement | 2-4 hours | ⏸️ Planned |

**Total Phase 1-4**: ~10-15 minutes (after current run completes)

## Key Implementation Details

### Jet Properties Computed

1. **n_constituents**: Number of particles in jet
2. **jet_pt**: Total transverse momentum
3. **jet_mass**: Invariant mass
4. **leading_pt_frac**: Leading particle pT / total jet pT
5. **pt_dispersion**: Standard deviation of constituent pT
6. **angular_width**: pT-weighted average ΔR from jet center
7. **girth**: Same as angular width (standard substructure variable)
8. **ptD**: Σ(pT_i²) / (Σ pT_i)² - measures pT concentration

### Statistical Tests

- **Mann-Whitney U**: Non-parametric test for distribution differences
- **Cohen's d**: Effect size measure (small: 0.2, medium: 0.5, large: 0.8)
- **p-value threshold**: 0.05 for significance

### Data Flow

```
Original Dataset (100,000 jets)
    ↓
Sequential/Random Sampling
    ↓
LLM Analysis + Tracking
    ↓
Easy/Hard Classification
    ├── Easy Jet Indices
    └── Hard Jet Indices
    ↓
Property Computation
    ├── Easy Jet Properties
    └── Hard Jet Properties
    ↓
Statistical Comparison
    ↓
Validation Experiments
    ├── 100 Easy Jets → AUC_easy
    ├── 100 Hard Jets → AUC_hard
    └── 50+50 Mixed → AUC_mixed
```

## Files Created/Modified

### Modified
- `scripts/analyze_llm_templates.py` - Added jet index tracking

### New Scripts
- `scripts/analyze_easy_hard_jets.py` - Property analysis
- `scripts/validate_easy_hard_hypothesis.py` - Validation experiments
- `scripts/run_easy_hard_investigation.sh` - Complete pipeline

### Documentation
- `notes/2025-10-11-easy-hard-jets-investigation-plan.md` - Detailed plan
- `notes/2025-10-11-implementation-summary.md` - This file
- `results/auc_vs_njets.{png,pdf}` - Sample size dependence plots

## Questions This Will Answer

1. **Are small samples biased?**
   → Compare AUC on 100 easy vs 100 random jets

2. **What makes jets "hard"?**
   → Statistical analysis of property differences

3. **Can we predict difficulty?**
   → Use properties to select easy/hard subsets

4. **How to improve templates?**
   → Target specific challenges identified in hard jets

5. **Is the downward trend real?**
   → Understand if it's due to sample composition or true performance

## Monitoring Progress

Check if analysis is complete:
```bash
ls -lh results/llm_analysis_reference_tracked.json
```

Monitor background job:
```bash
# Check if still running (will show process if running)
ps aux | grep analyze_llm_templates

# Or check output directory
watch -n 5 'ls -lh results/*.json | tail -5'
```

---

**Next**: Wait for reference analysis to complete, then proceed with property analysis and validation experiments.
