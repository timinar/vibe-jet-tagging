# Vibe Jet Tagging

A research project evaluating reasoning LLMs on particle physics jet tagging tasks, specifically quark/gluon discrimination.

> **New to the project?** See [installing UV](#install-uv) setup instructions at the end of this README.

## 🚀 Quick Start

We use `uv` (the modern Python package manager) throughout the project for dependency management and running code (see [installing UV](#install-uv)).

**First time setup:**
```bash
# 1. Sync dependencies
uv sync

# 2. Run scripts
uv run python scripts/your_script.py
```

**After first setup:**
```bash
# Sync dependencies (run after pulling changes)
uv sync

# Run scripts
uv run python scripts/your_script.py
```

## Common Commands

```bash
# Run scripts
uv run python scripts/your_script.py

# Testing
uv run pytest                              # All tests
uv run pytest --cov=src/vibe_jet_tagging   # With coverage

# Dependencies
uv add package-name              # Add new package (updates pyproject.toml)
uv add --dev dev-tool            # Add dev dependency
uv sync                          # Install/update all dependencies
```

## Project Overview

We evaluate reasoning LLMs on jet-tagging tasks using particle physics data. Jet tagging involves classifying jets of particles produced in high-energy collisionsspecifically distinguishing between quark-initiated and gluon-initiated jets based on their properties.

## Contributing

**Quick workflow:**
```bash
git checkout main && git pull && uv sync    # Start with latest
git checkout -b experiment/your-idea         # Create branch
# ... make changes ...
uv run pytest                                # Test
git commit -m "Add feature"                  # Commit
git push origin experiment/your-idea         # Push & create PR
```

---

## Setup Instructions

### Install UV


```bash
# Mac/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal or run:
source $HOME/.local/bin/env
```

Then clone and setup:
```bash
git clone https://github.com/yourusername/vibe-jet-tagging.git
cd vibe-jet-tagging
uv sync
```

