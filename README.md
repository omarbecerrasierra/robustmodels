# robustmodels

[![PyPI](https://img.shields.io/pypi/v/robustmodels)](https://pypi.org/project/robustmodels/)
[![Docs](https://github.com/omarbecerrasierra/robustmodels/actions/workflows/docs.yml/badge.svg)](https://omarbecerrasierra.github.io/robustmodels/)
[![CI](https://github.com/omarbecerrasierra/robustmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/omarbecerrasierra/robustmodels/actions/workflows/ci.yml)

Robust location, scale and covariance estimators for Python. Early development:
the API may change before version 1.0.

## Install

```bash
pip install robustmodels
```

Requires Python 3.10+, NumPy, SciPy and scikit-learn.

## Use

```python
from robustmodels import location, scale

x = [1, 2, 2, 3, 4, 100]
location.hodges_lehmann(x)
scale.mad(x)
scale.qn(x)
```

Covariance estimators follow the scikit-learn interface:

```python
import numpy as np
from robustmodels.covariance import MCD

X = np.random.default_rng(0).normal(size=(100, 3))
X[:10] += 8

model = MCD(random_state=0).fit(X)
model.location_
model.covariance_
model.mahalanobis(X)  # squared distances
model.support_  # observations kept after reweighting
```

## Estimators

| Module | Contents |
| --- | --- |
| `robustmodels.location` | `hodges_lehmann`, `m_location`, `tau_location`, `trimmed_mean`, `winsorized_mean` |
| `robustmodels.scale` | `mad`, `qn`, `sn`, `pn`, `tau_scale`, `m_scale`; skewness and tail weight: `medcouple`, `lmc`, `rmc` |
| `robustmodels.covariance` | `MCD`, `MRCD`, `MVE`, `OGK`, `SEstimator`, `MMEstimator`; correction factors: `h_alpha_n`, `mcd_consistency`, `mcd_small_sample` |

The [documentation](https://omarbecerrasierra.github.io/robustmodels/) explains how
to choose and use each estimator and documents every function and class. The same
information is in the docstrings: `help(robustmodels.scale.qn)`.

Univariate functions accept finite, one-dimensional samples. Empty samples return
`NaN`. A single observation has zero scale. Covariance estimators accept matrices
with observations in rows. MRCD supports more variables than observations and
uses an identity regularization target. MVE requires `alpha < 1`.
MRCD needs an explicit positive `rho` when a starting subset has zero scatter.

Qn, Sn and Pn include finite-sample corrections by default. Pass
`finite_correction=False` to disable them. Pn uses the factors from
[Tarr (2014), Appendix B.3](https://www.researchgate.net/publication/267927753_Quantile_Based_Estimation_of_Scale_and_Dependence).

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Build and check the distributions:

```bash
uv build
uvx twine check --strict dist/*
```

Run the docstring examples and build the documentation:

```bash
uv run pytest --doctest-modules --pyargs robustmodels
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

CI runs the tests against the installed wheel on Python 3.10, 3.13 and 3.14.
The Python 3.10 job uses the minimum supported NumPy, SciPy and scikit-learn versions.

The examples in `docs/` and in the docstrings run as tests. The tests cover estimator definitions, statistical properties, edge cases,
agreement with R's robustbase and rrcov, and
[scikit-learn compatibility](https://scikit-learn.org/stable/developers/develop.html).

Code lives in `src/robustmodels`:

| File | Contents | Internal dependencies |
| --- | --- | --- |
| `location.py` | Location estimators | `scale.py`, `_validation.py` |
| `scale.py` | Scale, skewness, correction factors and pairwise selection | `_validation.py` |
| `covariance.py` | Covariance estimators and their fitting steps | `scale.py`, `_validation.py` |
| `_validation.py` | Input and parameter checks | None |

Each method and its steps stay together. Auditing an estimator requires its file
and at most two supporting files. `__init__.py` exposes the three public modules.

### Release

1. Set `version` in `pyproject.toml`, run `uv lock`, and replace "unreleased"
   in [CHANGELOG.md](https://github.com/omarbecerrasierra/robustmodels/blob/main/CHANGELOG.md)
   with the date.
2. Merge into `main`, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. CI tests, builds and, after approval of the `pypi` environment, publishes to
   PyPI through trusted publishing. The tag must match the package version.

## License

BSD 3-Clause; see [LICENSE](https://github.com/omarbecerrasierra/robustmodels/blob/main/LICENSE).
