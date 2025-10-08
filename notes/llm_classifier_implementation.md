# LLMClassifier Implementation

## Overview
Implemented a complete LLM-based classifier for quark/gluon jet tagging using OpenRouter API.

## Components Created

### 1. Core Classifier (`src/vibe_jet_tagging/llm_classifier.py`)
- `LLMClassifier` class that inherits from base `Classifier`
- Uses OpenRouter API (OpenAI-compatible) for LLM inference
- Zero-shot classification (no training required)
- Configurable model, template, and data format

### 2. Data Formatters (`src/vibe_jet_tagging/utils/formatters.py`)
- `format_jet_as_list()`: Simple numbered list format
- `format_jet_as_yaml()`: Structured YAML format
- `format_jet_as_table()`: Markdown table format
- `load_template()`: Load prompt templates from files
- `fill_template()`: Fill templates with formatted jet data

### 3. Prompt Templates (`templates/`)
- `simple_list.txt`: Basic particle list format
- `structured_yaml.txt`: YAML with physics context
- `table_format.txt`: Table with classification hints

### 4. Test Notebook (`notebooks/test_llm_classifier.ipynb`)
- Load quark/gluon jet data
- Initialize LLMClassifier
- Test single jet prediction
- Run on 100 jets and compute AUC score
- Confusion matrix and performance metrics

## Usage

### Setup
```python
from vibe_jet_tagging import LLMClassifier

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Initialize classifier
clf = LLMClassifier(
    model_name="anthropic/claude-3.5-sonnet",
    template_name="simple_list",
    format_type="list",
    templates_dir="templates"
)

# Fit (no-op for zero-shot)
clf.fit([], [])
```

### Prediction
```python
# Single jet
prediction = clf.predict([jet])[0]  # Returns 0 or 1

# Multiple jets
predictions = clf.predict(X_test)  # Returns list of 0s and 1s
```

## Configuration Options

### Models
Any OpenRouter-compatible model:
- `"anthropic/claude-3.5-sonnet"`
- `"anthropic/claude-3-opus"`
- `"openai/gpt-4"`
- `"google/gemini-pro"`
- etc.

### Templates
- `"simple_list"` - Minimal formatting
- `"structured_yaml"` - Structured with context
- `"table_format"` - Table view with hints

### Format Types
- `"list"` - Numbered list format
- `"yaml"` - YAML structure
- `"table"` - Markdown table

## Dependencies Added
- `openai` - For OpenRouter API calls
- `scikit-learn` - For AUC score and metrics
- `ipykernel` - For Jupyter notebooks
- `tqdm` - For progress bars

## Next Steps
1. Run the notebook with actual API key
2. Test different templates and models
3. Compare performance across configurations
4. Add few-shot examples to improve accuracy
5. Implement baseline classifiers for comparison

