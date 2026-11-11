# robustmodels

**Robust statistics for Python, from individual measurements to multivariate data.**

[![PyPI version](https://img.shields.io/pypi/v/robustmodels)](https://pypi.org/project/robustmodels/)
[![Python versions](https://img.shields.io/pypi/pyversions/robustmodels)](https://pypi.org/project/robustmodels/)
[![CI](https://github.com/omarbecerrasierra/robustmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/omarbecerrasierra/robustmodels/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/omarbecerrasierra/robustmodels/graph/badge.svg)](https://codecov.io/gh/omarbecerrasierra/robustmodels)
[![Downloads](https://img.shields.io/pypi/dm/robustmodels)](https://pypistats.org/packages/robustmodels)
[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/omarbecerrasierra/robustmodels/blob/main/LICENSE)

[User guide](https://omarbecerrasierra.github.io/robustmodels/)
· [Method references](https://omarbecerrasierra.github.io/robustmodels/references.html)
· [Source code](https://github.com/omarbecerrasierra/robustmodels)
· [Report an issue](https://github.com/omarbecerrasierra/robustmodels/issues)

A few extreme observations can pull a mean away from the bulk of your data,
inflate a standard deviation, or distort a covariance matrix. **robustmodels**
brings established robust estimators into a consistent Python API so you can
compare summaries, estimate multivariate structure, and investigate unusual
observations.

- **Summarize a sample:** robust location, scale, skewness and tail-weight measures.
- **Model multivariate data:** six location and covariance estimators, including
  regularized covariance for data with more variables than observations.
- **Work with familiar tools:** NumPy-compatible inputs and the scikit-learn
  estimator interface for covariance models.
- **Inspect the methods:** documented formulas, scientific references, numerical
  conventions and executable examples.

## Installation

```bash
python -m pip install robustmodels
```

Requires **Python 3.10+**. NumPy, SciPy and scikit-learn are installed as dependencies.
CI tests Python 3.10–3.14 on Linux, Windows and macOS.

## Start with one sample

Compare ordinary and robust summaries of a sample containing one extreme value:

```python
import numpy as np
from robustmodels import location, scale

x = [1, 2, 2, 3, 4, 100]

print(f"Mean:           {np.mean(x):.2f}")
print(f"Hodges–Lehmann:  {location.hodges_lehmann(x):.2f}")
print(f"Standard dev.:  {np.std(x, ddof=1):.2f}")
print(f"Qn scale:       {scale.qn(x):.2f}")
```

```text
Mean:           18.67
Hodges–Lehmann:  3.00
Standard dev.:  39.86
Qn scale:       1.36
```

The robust summaries remain close to the location and spread of the five values
between 1 and 4. Qn is calibrated to estimate the normal standard deviation and
includes a finite-sample correction by default.

You can also use `location.m_location(x)`, `scale.mad(x)` or
`scale.medcouple(x)` to estimate a robust center, scale or skewness. The
[location guide](https://omarbecerrasierra.github.io/robustmodels/location.html)
and [scale guide](https://omarbecerrasierra.github.io/robustmodels/scale.html)
explain their tuning and tradeoffs.

## Explore multivariate outliers

Fit a Minimum Covariance Determinant model and compute squared robust
Mahalanobis distances. This example adds a shift to ten observations:

```python
import numpy as np
from scipy.stats import chi2
from robustmodels.covariance import MCD

X = np.random.default_rng(0).normal(size=(100, 2))
X[:10] += 10

model = MCD(random_state=0).fit(X)
distances = model.mahalanobis(X)
cutoff = chi2.ppf(0.975, df=X.shape[1])

print("Robust center:", model.location_.round(2).tolist())
print("Rows to inspect:", np.flatnonzero(distances > cutoff).tolist())
```

```text
Robust center: [0.01, 0.1]
Rows to inspect: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 69]
```

The ten shifted observations are flagged, along with one unshifted observation
in the tail. Treat these as candidates for investigation. The chi-square cutoff
is a reference under a multivariate normal model, not an exact finite-sample
test or a guaranteed false-positive rate.

The fitted model exposes `location_`, `covariance_`, `precision_` and `support_`.
Use `mahalanobis(new_X)` to score new observations; it returns **squared**
distances. Covariance estimators support `get_params`, `set_params` and
`sklearn.base.clone`.

## Choose an estimator

| Your task | A starting point | What to consider |
| --- | --- | --- |
| Estimate the center of one sample | `location.hodges_lehmann` | No tuning parameter; based on pairwise averages. |
| Estimate spread with resistance to outliers | `scale.qn` or `scale.mad` | Qn uses pairwise differences; MAD uses deviations from a center. |
| Describe asymmetry | `scale.medcouple` | Robust skewness; `lmc` and `rmc` describe the tails. |
| Estimate covariance when observations outnumber variables | `MCD` | Supports random or deterministic starts. |
| Estimate covariance when variables outnumber observations | `MRCD` | Regularizes toward an identity target in standardized coordinates. |
| Explore other multivariate approaches | `OGK`, `MVE`, `SEstimator`, `MMEstimator` | Different robustness, efficiency and computational tradeoffs. |

For example, regularized covariance can be fitted to a wide matrix:

```python
import numpy as np
from robustmodels.covariance import MRCD

wide = np.random.default_rng(0).normal(size=(20, 50))
regularized = MRCD().fit(wide)
print(regularized.covariance_.shape)
```

```text
(50, 50)
```

See the [covariance guide](https://omarbecerrasierra.github.io/robustmodels/covariance.html)
for sample-size requirements, parameters and worked examples.

## Available methods

| Module | Public estimators |
| --- | --- |
| `robustmodels.location` | `hodges_lehmann`, `m_location`, `tau_location`, `trimmed_mean`, `winsorized_mean` |
| `robustmodels.scale` | `mad`, `qn`, `sn`, `pn`, `tau_scale`, `m_scale`, `medcouple`, `lmc`, `rmc` |
| `robustmodels.covariance` | `MCD`, `MRCD`, `MVE`, `OGK`, `SEstimator`, `MMEstimator` |

The covariance module also exposes `h_alpha_n`, `mcd_consistency` and
`mcd_small_sample`. Each public method has a docstring with parameters,
return values, examples and references.

## Conventions and scope

- Univariate estimators take finite, one-dimensional samples. Empty samples
  return `NaN`; a single observation has zero scale. Handle missing and infinite
  values before fitting.
- Covariance models take matrices with **observations in rows** and variables
  in columns. Set `random_state` for methods that use random starts.
- Qn, Sn and Pn apply finite-sample corrections by default. Use
  `finite_correction=False` when you need the asymptotic calibration.
- Robustness depends on the estimator, sample size, dimension and data. No method
  in this package is immune to arbitrary contamination. Multivariate subset
  searches are approximations and do not certify global optima.
- The package is in **early development**. Its API may change before version 1.0.

The [method references](https://omarbecerrasierra.github.io/robustmodels/references.html)
connect the implementations to their scientific sources and explain differences
from R conventions. Tests cover statistical properties, edge cases, reference
comparisons with R's robustbase and rrcov, and scikit-learn compatibility.

## Documentation, contributions and citation

- **Learn:** the [user guide](https://omarbecerrasierra.github.io/robustmodels/)
  includes the API reference and examples tested alongside the package.
- **Contribute:** see the [development guide](https://github.com/omarbecerrasierra/robustmodels/blob/main/CONTRIBUTING.md)
  for setup, checks and release instructions. Bug reports and reproducible
  examples are welcome in [Issues](https://github.com/omarbecerrasierra/robustmodels/issues).
- **Follow changes:** read the [changelog](https://github.com/omarbecerrasierra/robustmodels/blob/main/CHANGELOG.md).
- **Cite:** use the metadata in [CITATION.cff](https://github.com/omarbecerrasierra/robustmodels/blob/main/CITATION.cff)
  or GitHub's “Cite this repository” option.

Released under the [BSD 3-Clause license](https://github.com/omarbecerrasierra/robustmodels/blob/main/LICENSE).
