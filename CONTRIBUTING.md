# Contributing to robustmodels

For bugs, include a small reproducible example, the expected result, the actual
result, and your Python and dependency versions. For changes to an estimator,
include a scientific reference and explain the numerical convention being used.

## Development setup

```bash
uv sync --locked --group docs
```

Run the tests and style checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Measure line and branch coverage, which CI requires to stay at 100%:

```bash
uv run pytest --cov
```

Build and check the distributions:

```bash
uv build
uvx twine check --strict dist/*
```

Run the docstring examples and build the documentation:

```bash
uv run pytest --doctest-modules --pyargs robustmodels
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

CI tests the installed wheel on Python 3.10–3.14 on Linux, Windows and macOS.
Python 3.10 jobs use the minimum supported NumPy, SciPy and scikit-learn versions.
The examples in `docs/` and the docstrings run as tests. The test suite covers
estimator definitions, statistical properties, edge cases, comparisons with R's
robustbase and rrcov, and scikit-learn compatibility. A separate CI job checks
coverage and reports it to Codecov. Mark a line with `# pragma: no cover` only when
it cannot be reached, and say why in the comment.

## Code organization

| File in `src/robustmodels` | Contents | Internal dependencies |
| --- | --- | --- |
| `location.py` | Location estimators | `scale.py`, `_validation.py` |
| `scale.py` | Scale, skewness, correction factors and pairwise selection | `_validation.py` |
| `covariance.py` | Covariance estimators and their fitting steps | `scale.py`, `_validation.py` |
| `regression.py` | Regression estimators and their fitting steps | `covariance.py`, `scale.py`, `_validation.py` |
| `_validation.py` | Input checks and the random generator helper | None |

Each method and its steps stay together. `__init__.py` exposes the public
modules. Update the relevant docstrings and user guide when changing public
behavior, including numerical limits and differences from reference software.

## Release checklist

1. Set the version in `pyproject.toml` and run `uv lock`. Date the changelog entry
   and update the version and release date in `CITATION.cff`.
2. Validate the README examples and rendered package description, run the checks
   above, and commit the release changes to `main`.
3. Tag that commit with the matching version, for example `v0.1.1`, and push the
   tag. CI tests and builds the release before publishing.
4. Review and approve deployment to the `pypi` environment. Trusted publishing
   uploads the distributions without a stored PyPI token.
5. Verify the version and description on PyPI, then create the matching GitHub
   release using the changelog entry.

PyPI descriptions come from `README.md` in the uploaded distribution. Updating
the README on GitHub alone does not update an existing PyPI release. Use absolute
URLs for links and check both the wheel and source distribution with Twine.
