# robustmodels

Robust location, scale and covariance estimators for Python. Early development.

Requires Python 3.10+, NumPy, SciPy and scikit-learn.

## Install

From a local checkout:

```bash
pip install .
```

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

- `location`: Hodges–Lehmann, M-location, tau location, trimmed and winsorized means.
- `scale`: MAD, Qn, Sn, Pn, tau scale, M-scale and medcouple.
- `covariance`: MCD, MRCD, MVE, OGK, S and MM.

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

CI runs the tests against the installed wheel on Python 3.10, 3.13 and 3.14.
The Python 3.10 job uses the minimum supported NumPy, SciPy and scikit-learn versions.

The tests cover estimator definitions, statistical properties, edge cases and
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

Mathematical definitions and paper references are kept beside each implementation.

## License

BSD 3-Clause.
