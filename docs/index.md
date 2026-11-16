# robustmodels

Robust statistics for Python: estimators of location, scale, covariance and
linear and nonlinear regression, generalized linear models, principal
components, discriminant analysis, statistical depth and outlier detectors, that stay reliable when part of the data are outliers.

```bash
pip install robustmodels
```

Requires Python 3.10+, NumPy, SciPy, scikit-learn and Matplotlib.

## Quick start

One gross outlier ruins the mean and the standard deviation, but not their
robust counterparts:

```pycon
>>> import numpy as np
>>> from robustmodels import location, scale
>>> x = [1, 2, 2, 3, 4, 100]
>>> round(float(np.mean(x)), 2), round(float(np.std(x, ddof=1)), 2)
(18.67, 39.86)
>>> location.hodges_lehmann(x), round(scale.qn(x), 2)
(3.0, 1.36)

```

The covariance estimators are scikit-learn estimators:

```pycon
>>> from robustmodels.covariance import MCD
>>> X = np.random.default_rng(0).normal(size=(100, 2))
>>> X[:10] += 10  # ten outliers
>>> mcd = MCD(random_state=0).fit(X)
>>> mcd.location_.round(2).tolist()
[0.01, 0.1]
>>> mcd.covariance_.round(2).tolist()
[[0.93, 0.15], [0.15, 1.06]]

```

## Contents

| Page | What it covers |
| --- | --- |
| [Location](location.md) | Hodges-Lehmann, M-estimators, tau location, trimmed and winsorized means |
| [Scale and skewness](scale.md) | MAD, Qn, Sn, Pn, tau scale, M-scale, medcouple and tail weights |
| [Method references](references.md) | Scientific sources, formulas and R conventions |
| [Covariance](covariance.md) | MCD, MRCD, MVE, OGK, Stahel-Donoho, S- and MM-estimators; outlier detection |
| [Multivariate methods](multivariate.md) | Robust PCA (ROBPCA, projection pursuit), LDA and QDA, Hotelling's T² and MANOVA |
| [Regression](regression.md) | MM, S, LTS and nonlinear M regression |
| [Generalized linear models](glm.md) | Robust logistic and Poisson models, inference and numerical conventions |
| [Depth](depth.md) | Halfspace and regression depth, Tukey median and bagplot |
| [Outlier detection](outliers.md) | Stahel-Donoho, adjusted and directional outlyingness; projection depth |
| [Diagnostic plots](plots.md) | Regression and PCA outlier maps, quantile plots, distance-distance plots, tolerance ellipses, scree plots, biplots, classification regions, adjusted boxplots, bagplots and outlyingness |
| [Validation](validation.md) | Definitions, pre-release checks and documented differences from R |

Each page explains how to choose and use the estimators, then documents every
function and class: definition, parameters, return value, references and an
example.

## Conventions

- Univariate functions take a one-dimensional sample of finite values and return
  a Python `float`. `NaN` or infinite values raise `ValueError`.
- An empty sample returns `nan`; a single observation has zero scale.
- Covariance estimators take a matrix with observations in rows and variables
  in columns.
- Invalid parameters raise `ValueError` with the name of the parameter.
- The **breakdown point** is the largest fraction of arbitrary outliers an
  estimator tolerates. The **Gaussian efficiency** compares its variance with
  that of the classical estimator on clean normal data.

```{toctree}
:hidden:

location
scale
covariance
multivariate
regression
glm
depth
outliers
plots
validation
from-r
references
roadmap
changelog
```
