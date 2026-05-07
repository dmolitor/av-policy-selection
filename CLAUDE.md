# CLAUDE.md

## Essential Project Background

`av-policy-selection` is a Python research project implementing anytime-valid statistical methods for optimal policy identification and off-policy inference in contextual bandits.

There are two documents that are the critical source of truth on which all code will be based:
- `resources/anytime-valid-off-policy-inference-for-contextual-bandits.tex` is the paper that
introduces the primary methodological tools. All definitions and essential background are contained
in this paper.
- `resources/anytime-valid-optimal-policy-identification.tex` is our proposed extension built on
top of the initial paper.

All the code implemented in this project should be linked directly to these papers
and corresponding references to the papers should be included in the documentation
when helpful.

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
```

## Code Architecture and Instructions

The package lives in `src/av_policy_selection/`.

IMPORTANT: All code we add should have corresponding unit tests developed. These unit tests
should meet one of two standards: (1) if the code does a non-deterministic process (i.e. we can't
just confirm that it's output matches a given quantity) the unit tests should confirm that
it's behavior matches expected behavior; (2) if the code is deterministic, the unit tests 
should confirm that it matches expected behavior AND that the output is as expected.

After adding new code, you MUST ALWAYS run all unit tests and fix any resulting errors before
proceeding.

## Git instructions

Never use git unless explicitly instructed/asked by me. When instructed,
all changes should be made to the `dev` branch unless I explicity tell you
another branch.