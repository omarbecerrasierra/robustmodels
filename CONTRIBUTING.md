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

CI tests the installed wheel on Python 3.10–3.15 on Linux, Windows and macOS.
Python 3.10 jobs use the minimum supported NumPy, SciPy and scikit-learn versions.
The examples in `docs/` and the docstrings run as tests. The test suite covers
estimator definitions, statistical properties, exact small problems, numerical
derivatives, edge cases and scikit-learn compatibility. Frozen outputs from
other software are supplementary regression checks with recorded provenance.
They must not replace independent verification of the mathematical definition. A separate CI job checks
coverage and reports it to Codecov. Mark a line with `# pragma: no cover` only when
it cannot be reached, and say why in the comment.

## Code organization

| File in `src/robustmodels` | Contents | Internal dependencies |
| --- | --- | --- |
| `location.py` | Location estimators | `scale.py`, `_validation.py` |
| `scale.py` | Scale, skewness, correction factors and pairwise selection | `_validation.py` |
| `covariance.py` | Covariance estimators and their fitting steps | `location.py`, `outliers.py`, `scale.py`, `_validation.py` |
| `multivariate.py` | Principal components, discriminant analysis and tests on mean vectors | `covariance.py`, `_validation.py` |
| `cluster.py` | Trimmed clustering | `_validation.py` |
| `regression.py` | Linear and nonlinear regression and inference | `covariance.py`, `scale.py`, `_validation.py` |
| `glm.py` | Robust logistic and Poisson models and inference | `regression.py`, `covariance.py`, `scale.py`, `_validation.py` |
| `rank.py` | Rank regression and conditional rank tests | `regression.py`, `scale.py`, `_validation.py` |
| `density.py` | Kernel density estimation, bandwidths, derivatives, regions and tests | `_validation.py` |
| `shape.py` | Log-concave density and score matching regression | `rank.py`, `regression.py`, `scale.py`, `_validation.py` |
| `depth.py` | Halfspace and regression depth, Tukey median and bagplot statistics | `outliers.py`, `_validation.py` |
| `outliers.py` | Outlier detectors by projection outlyingness | `scale.py`, `_validation.py` |
| `plots.py` | Matplotlib diagnostics, adjusted boxplots and bagplots | `covariance.py`, `depth.py`, `outliers.py`, `scale.py`, `_validation.py` |
| `datasets.py` | Classic data sets, stored as CSV files in `data/` | None |
| `_validation.py` | Input checks, random generator and numerical helpers | None |

Each method and its steps stay together. `__init__.py` exposes the public
modules. Update the relevant docstrings and user guide when changing public
behavior, including numerical limits and differences from reference software.

Use public NumPy, SciPy and scikit-learn functions for operations they already
provide. Reuse an existing estimator or formula from robustmodels when it
matches the method's definition. Check divisors, ties, weights and numerical
limits before substituting an implementation.
Use named intermediate values when they make it easier to follow a formula
or check the order of an algorithm's steps.

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
