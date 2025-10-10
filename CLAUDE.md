# Claude Development Notes

This file contains context and conventions for AI-assisted development.

## Project Overview

This project evaluates reasoning LLMs on particle physics jet tagging tasks, specifically quark/gluon discrimination.

## Code Style

- Follow PEP 8 conventions
- Type hints are encouraged (project uses `py.typed`)

## Project Structure

- `src/vibe_jet_tagging/`: Main package code
- `tests/`: Test suite (pytest)
- `scripts/`: Analysis scripts and experiments

## Development Workflow

- Use `uv` for all dependency management
- Run tests with `uv run pytest`
- Add dependencies via `uv add package-name`

## Testing

- Write tests for all new functionality
- Aim for high test coverage
- Tests should be in `tests/` mirroring the `src/` structure

## Documentation and Notes

- Put documentation and notes in `notes/` directory
- Start note filenames with the date in YYYY-MM-DD format (e.g., `2025-10-10-async-fixes.md`)
- Keep the project root clean - avoid creating documentation files in root
