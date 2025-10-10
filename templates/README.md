# Prompt Templates

This directory contains prompt templates for the LLMClassifier.

## Template Overview

| # | Template Name | Category | Data Type | Features | Extractor | Physics Hints | Token Efficiency | Use Case |
|---|---------------|----------|-----------|----------|-----------|---------------|------------------|----------|
| 1 | `simple_list.txt` | Basic | Raw particles | None | None | None | Low | Pure zero-shot baseline |
| 2 | `structured_yaml.txt` | Basic | Raw particles (YAML) | None | None | Minimal | Low | Alternative format test |
| 3 | `table_format.txt` | Basic | Raw particles (table) | None | None | Brief | Low | Tabular format test |
| 4 | `with_summary_stats.txt` | Informed | Raw particles | None | None | Dataset stats | Low | Contextualized zero-shot |
| 5 | `with_optimal_cut.txt` | Informed | Raw particles | None | None | Decision rule (38 cut) | Low | Test heuristic application |
| 6 | `with_engineered_features.txt` | Informed | Raw particles | None | None | Feature guidance | Low | Expert system approach |
| 7 | `features_basic.txt` | Feature-only | Multiplicity | Multiplicity | BasicExtractor | Decision rule | **High** | Most efficient |
| 8 | `features_kinematic.txt` | Feature-only | Kinematic | Mult + pT stats | KinematicExtractor | Physics patterns | **High** | Balanced efficiency |
| 9 | `features_full.txt` | Feature-only | All features | 8 features | FullExtractor | Comprehensive | **High** | Maximum feature info |
| 10 | `hybrid_basic.txt` | Hybrid | Particles + features | Multiplicity | BasicExtractor | Decision rule | Medium | Simple rule + verification |
| 11 | `hybrid_kinematic.txt` | Hybrid | Particles + features | Mult + pT stats | KinematicExtractor | Physics patterns | Medium | **Recommended** |
| 12 | `hybrid_full.txt` | Hybrid | Particles + features | 8 features | FullExtractor | Comprehensive | Low | Maximum information |

### Category Definitions

- **Basic**: Minimal context, raw particle data only
- **Informed**: Raw particles + increasing physics knowledge
- **Feature-only**: Only engineered features, no raw particles (token efficient)
- **Hybrid**: Both engineered features AND raw particles (best of both worlds)

### Feature Descriptions

| Feature | Description | Quark (avg) | Gluon (avg) | Importance |
|---------|-------------|-------------|-------------|------------|
| Multiplicity | Number of particles | ~33 | ~53 | ⭐⭐⭐⭐⭐ |
| Mean pT | Average transverse momentum | ~15.7 GeV | ~9.8 GeV | ⭐⭐⭐⭐ |
| Std pT | pT spread | - | - | ⭐⭐⭐ |
| Median pT | Median transverse momentum | - | - | ⭐⭐ |
| Max pT | Leading particle pT | - | - | ⭐⭐ |
| Leading pT fraction | Fraction in leading particle | Higher | Lower | ⭐⭐⭐ |
| Top-3 pT fraction | Fraction in top 3 particles | Higher | Lower | ⭐⭐⭐ |
| Top-5 pT fraction | Fraction in top 5 particles | Higher | Lower | ⭐⭐ |

### Template Placeholders

Templates can use the following placeholders (automatically replaced by LLMClassifier):

| Placeholder | Replacement | Example |
|-------------|-------------|---------|
| `{{jet_particles}}` | List-formatted particle data | `Particle 1: pt=2.5 GeV, ...` |
| `{{jet_yaml}}` | YAML-formatted particle data | `particles:\n  - pt: 2.5` |
| `{{jet_table}}` | Markdown table of particles | `\| pt \| y \| phi \|` |
| `{{multiplicity}}` | Number of particles | `18` |
| `{{jet_features}}` | Formatted feature set | `Multiplicity: 18\nMean pT: 27.8 GeV\n...` |

## Quick Selection Guide

### By Goal

| Goal | Recommended Templates |
|------|----------------------|
| Baseline performance | `simple_list` |
| Token efficiency | `features_basic`, `features_kinematic` |
| Best performance | `hybrid_kinematic`, `hybrid_full` |
| Test feature value | Compare `features_*` vs `hybrid_*` |
| Test physics knowledge | `with_summary_stats`, `with_optimal_cut` |

### By Expected Performance

Based on baseline analysis (20,000 test jets):

| Model | AUC | Notes |
|-------|-----|-------|
| Random guess | 0.50 | - |
| Simple multiplicity cut (>38) | 0.84 | Single feature |
| Logistic regression (mult only) | 0.84 | Single feature |
| Logistic regression (8 features) | 0.85 | Hand-crafted features |
| XGBoost (raw particles) | 0.86 | Best baseline |
| **Target for LLM** | **≥0.85** | Should match/exceed feature-based |

## Auto-Detection System

The LLMClassifier automatically detects required features from template placeholders:

```python
# Example 1: Multiplicity only
template = "Jets with {{multiplicity}} particles..."
# → Auto-selects BasicExtractor

# Example 2: Multiple features
template = "Here are the features:\n{{jet_features}}"
# → Auto-selects FullExtractor (or KinematicExtractor based on context)

# Example 3: Hybrid
template = "Features:\n{{jet_features}}\n\nParticles:\n{{jet_particles}}"
# → Extracts features AND includes raw particles
```

## Creating New Templates

To create a new template:

1. Create a `.txt` file in this directory
2. Write your prompt with placeholders (see table above)
3. Use in LLMClassifier:
   ```python
   clf = LLMClassifier(
       template_name="your_template_name",
       templates_dir="path/to/templates"
   )
   ```
4. The system will auto-detect required feature extractor

## Performance Tips

1. **Start simple**: Test `features_basic` first (fastest, cheapest)
2. **Compare systematically**: Raw → Features → Hybrid
3. **Cost vs Performance**: Feature-only templates save ~20-50% on tokens
4. **Reasoning budget**: Higher budgets (1000-4000 tokens) may help complex templates
