# Contributing

## Scope

This is a sanitized public research repository. Contributions should improve:
- code clarity;
- reproducibility;
- documentation;
- public-safe experiment packaging.

Do not contribute:
- private benchmark data;
- heavyweight checkpoints;
- manifests with sensitive absolute paths;
- screenshots or figures derived from non-public images.

## Development Workflow

1. Create a fresh environment.
2. Install with `python -m pip install -r requirements-dev.txt` and `python -m pip install -e .[dev]`.
3. Run `python -m unittest discover -s tests -v`.
4. Run `ruff check src tests`.
5. Keep documentation and public result summaries in sync with code changes.

## Pull Requests

Include:
- problem statement;
- files changed;
- tests run;
- whether any public-facing docs need updating;
- whether the change touches reproducibility or data-access assumptions.
