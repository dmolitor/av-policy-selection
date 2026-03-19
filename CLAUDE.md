# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`av-policy-selection` is a Python research project implementing anytime-valid statistical methods for optimal policy identification and off-policy inference in contextual bandits. The theoretical foundation is in `resources/` — two LaTeX papers covering doubly robust pseudo-outcomes, confidence sequences (CS), and stopping time analysis.

## Package Manager

This project uses `uv`. Use `uv` for all dependency and environment management:

```bash
uv sync           # Install dependencies
uv add <pkg>      # Add a dependency
uv run <cmd>      # Run a command in the project environment
```

## Common Commands

```bash
uv run python -c "import av_policy_selection"   # Verify package is importable
uv run pytest                                    # Run tests (once test suite exists)
uv run pytest tests/test_foo.py::test_bar       # Run a single test
uv run ruff check src/                          # Lint (if ruff is added)
uv run mypy src/                                # Type check (if mypy is added)
```

## Code Architecture

The package lives in `src/av_policy_selection/`. It is currently in early development with only a stub `hello()` function — the real implementation will follow the mathematical framework in `resources/`.

**Key concepts from the research papers (relevant when implementing):**
- **Doubly robust pseudo-outcomes**: Core estimator construction for off-policy evaluation
- **Confidence sequences (CS)**: Anytime-valid, time-uniform confidence bounds (not fixed-sample CIs)
- **Optimal policy set**: The set of policies whose value is within `ε` of the best policy
- **Stopping times**: Sequential decision rules for when to terminate policy selection

Python 3.14 is required (see `.python-version`).
