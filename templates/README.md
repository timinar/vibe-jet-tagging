# Prompt Templates

This directory contains prompt templates for the LLMClassifier.

## Available Templates

### 1. `simple_list.txt`
- Formats jet particles as a simple numbered list
- Each particle shows: pt, y, phi, and pid
- Minimal formatting, easy to read

### 2. `structured_yaml.txt`
- Formats jet data as YAML
- More structured representation
- Includes physics context about quark vs gluon differences

### 3. `table_format.txt`
- Formats particles as a markdown table
- Clean tabular view of particle properties
- Includes brief classification hints

## Template Placeholders

Templates can use the following placeholders:
- `{{jet_particles}}` - Replaced with list-formatted jet data
- `{{jet_yaml}}` - Replaced with YAML-formatted jet data
- `{{jet_table}}` - Replaced with table-formatted jet data

## Creating New Templates

To create a new template:
1. Create a `.txt` file in this directory
2. Write your prompt with placeholders
3. Use it in LLMClassifier with `template_name="your_template_name"`

