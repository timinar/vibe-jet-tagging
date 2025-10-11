# Easy vs Hard Jets Investigation Plan

**Date**: 2025-10-11
**Goal**: Understand why AUC decreases with sample size and identify characteristics that make jets "easy" or "hard" for LLM classification

## Background

Analysis of bootstrap results shows:
- **100 jets**: AUC ≈ 0.87 (CI: 0.81-0.93)
- **1000 jets**: AUC ≈ 0.75 (CI: 0.73-0.78)
- **3000 jets**: AUC ≈ 0.77 (CI: 0.75-0.78)

**Problem**: CIs do NOT overlap → statistically significant difference suggests the 100-jet sample is biased toward "easy" jets.

## Dataset Structure

- **Total jets**: 100,000 (from `qg_jets.npz`)
- **Shape**: X: (100000, 139, 4), y: (100000,)
- **Current sampling**: Sequential (indices 0 to num_jets-1)
- **Jet identification**: Can use array index as unique ID

## Investigation Plan

### Phase 1: Jet Identification and Tracking

**Objective**: Modify analysis pipeline to track which specific jets are classified correctly/incorrectly

**Tasks**:
1. ✅ Understand current sampling (done: uses indices 0:num_jets)
2. Add `jet_indices` field to analysis results
3. Save jet indices alongside predictions and true labels
4. Test on small sample to verify tracking works

**Implementation**:
- Modify `analyze_llm_templates.py` to include `jet_indices` in results
- Add `--random_sample` flag to allow random vs sequential sampling
- Add `--seed` parameter for reproducibility

### Phase 2: Define "Easy" vs "Hard" Jets

**Objective**: Classify jets based on LLM classification difficulty

**Method 1: Single-Run Classification**
- Run on N=1000 jets (large enough for statistical significance)
- Easy jet: `prediction == true_label`
- Hard jet: `prediction != true_label`
- **Pros**: Simple, direct
- **Cons**: Single prediction may have random errors

**Method 2: Bootstrap Consensus (RECOMMENDED)**
- Use bootstrap results (1000 resamples) from existing analysis
- For each jet i, count how many times it was correctly classified across bootstrap samples
- **Easy jets**: Correct in ≥80% of bootstrap samples (high consensus)
- **Hard jets**: Correct in ≤60% of bootstrap samples (low consensus)
- **Ambiguous**: 60-80% (exclude from targeted analysis)
- **Pros**: More robust, uses existing bootstrap data
- **Cons**: Requires bootstrap tracking of jet indices

**Method 3: Multi-Model Consensus**
- Run with multiple reasoning efforts (low, medium)
- Easy: Correct with both efforts
- Hard: Incorrect with both efforts
- **Pros**: Tests across different model configurations
- **Cons**: More expensive

**Choice**: Start with Method 1 (single-run), then enhance with Method 2 (bootstrap) if we modify tracking

### Phase 3: Property Analysis

**Objective**: Identify kinematic/substructure features that distinguish easy from hard jets

**Properties to Analyze**:

1. **Basic Kinematics**:
   - Jet pT (transverse momentum)
   - Jet mass
   - Number of constituents
   - Leading particle pT fraction

2. **Substructure Variables** (compute from constituents):
   - pT dispersion: σ(pT) among constituents
   - Angular width: mean ΔR from jet axis
   - Constituent multiplicity distribution
   - Energy distribution asymmetry

3. **Quark/Gluon Discriminants**:
   - Girth: Σ(pT_i × ΔR_i) / Σ(pT_i)
   - Les Houches angularity
   - pT^D: Σ(pT_i²) / (Σ pT_i)²

4. **Decision Boundary Distance** (if applicable):
   - For jets near quark/gluon boundary, classification should be harder

**Analysis Methods**:
- Distributions: Plot histograms of each property for easy vs hard jets
- Statistical tests: KS test, Mann-Whitney U test for differences
- Correlations: Which properties best separate easy/hard?
- Clustering: Can we identify distinct "difficulty classes"?

**Implementation**:
- Create `scripts/analyze_jet_properties.py`
- Compute all properties from raw constituent data
- Generate comparative plots
- Statistical summary table

### Phase 4: Hypothesis Validation

**Objective**: Validate that identified properties predict difficulty

**Experiments**:

**Experiment 1: Targeted Sampling**
- Select 100 easy jets (based on properties from Phase 3)
- Select 100 hard jets (based on properties from Phase 3)
- Run LLM analysis on each subset separately
- **Expected Result**:
  - Easy subset: AUC ≈ 0.85-0.90 (high, like the 100-jet sample)
  - Hard subset: AUC ≈ 0.55-0.65 (near random)
  - This would confirm that sample composition drives AUC variation

**Experiment 2: Stratified Sampling**
- Create balanced samples across difficulty spectrum
- E.g., 100 jets with 50% easy, 50% hard
- Compare to random 100 jets
- **Expected Result**: Controlled AUC based on mix

**Experiment 3: Cross-Validation by Difficulty**
- 5-fold CV within easy jets only
- 5-fold CV within hard jets only
- Compare variance
- **Expected Result**: Lower variance within each difficulty class

**Implementation**:
- Create `scripts/validate_easy_hard_hypothesis.py`
- Use jet selection based on Phase 3 findings
- Run systematic experiments
- Compare results to original 100/1000/3000 jet analyses

### Phase 5: Template Improvement

**Objective**: Improve prompt template to help LLM classify hard jets correctly

**Strategy**:
Based on Phase 3 findings, hypotheses:

**If hard jets have complex substructure**:
- Add explicit guidance about interpreting complex particle patterns
- Highlight key discriminative features (e.g., "wide angular distributions suggest gluon")
- Provide examples of complex cases

**If hard jets are near decision boundary**:
- Add uncertainty quantification to prompt
- Ask LLM to explain reasoning for borderline cases
- Provide calibration examples

**If hard jets have specific kinematic regions**:
- Add conditional logic in template
- E.g., "If jet has low multiplicity, focus on X; if high, focus on Y"

**Implementation**:
- Create `templates/with_optimal_cut_enhanced.md`
- Test on hard jets subset
- Compare AUC improvement
- Full re-evaluation on 1000 jets

**Success Metrics**:
- AUC improvement on hard jets: +0.05-0.10
- Reduced AUC variance across sample sizes
- More consistent performance at 100 vs 1000 jets

## Expected Outcomes

### Hypothesis Confirmation
If our hypothesis is correct:
1. Easy jets have distinct kinematic signatures (e.g., clear quark/gluon separation)
2. Hard jets are near decision boundary or have ambiguous features
3. Small N samples (100 jets) are biased toward easy jets due to sampling variance
4. Large N samples (1000+) include representative mix, showing true (lower) performance

### Template Improvements
- Identify specific guidance needed for hard cases
- Create enhanced template with better performance on hard jets
- Reduce dependence of AUC on sample size

### Scientific Understanding
- Characterize which quark/gluon jets are fundamentally harder to classify
- Understand LLM's decision-making process for jet tagging
- Identify limitations of current prompt engineering approach

## Timeline Estimate

- **Phase 1** (Tracking): 1-2 hours
- **Phase 2** (Classification): 1 hour
- **Phase 3** (Properties): 2-3 hours
- **Phase 4** (Validation): 2-3 hours (includes LLM runs)
- **Phase 5** (Improvement): 2-4 hours (iterative)

**Total**: ~8-13 hours

## Files to Create/Modify

**New Scripts**:
- `scripts/analyze_easy_hard_jets.py` - Main analysis script
- `scripts/compute_jet_properties.py` - Property calculations
- `scripts/validate_easy_hard_hypothesis.py` - Validation experiments
- `scripts/select_jets_by_difficulty.py` - Jet selection utilities

**Modified Scripts**:
- `scripts/analyze_llm_templates.py` - Add jet index tracking
- `scripts/bootstrap_llm_results.py` - Track indices in bootstrap (optional)

**New Templates**:
- `templates/with_optimal_cut_enhanced.md` - Improved template

**Documentation**:
- This plan (current file)
- Results summary (to be created after completion)

## Next Steps

1. Implement jet index tracking in analysis pipeline
2. Run reference analysis on 1000 jets with tracking enabled
3. Classify jets as easy/hard
4. Proceed with property analysis

---

**Note**: This plan is subject to revision based on findings at each phase.
