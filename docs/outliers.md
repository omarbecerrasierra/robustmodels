# Outlier detection by projection outlyingness and depth

Detectors that flag multivariate outliers by how far each observation lies from
the bulk of the data along many one-dimensional projections, in
`robustmodels.outliers`.

## Choosing a detector

| Class | Data | Notes |
| --- | --- | --- |
| `StahelDonohoOutlyingness` | Roughly symmetric | Median and MAD of each projection; fast |
| `AdjustedOutlyingness` | Skewed | Adjusted boxplot of each projection; finds outliers that skewness masks |

Both need more observations than variables, are affine invariant, and make no
assumption of elliptical shape. `covariance.MCD` with robust distances is an
alternative for elliptical data.

## Use

The detectors follow scikit-learn's outlier detection API, as
`sklearn.covariance.EllipticEnvelope`:

```pycon
>>> import numpy as np
>>> from robustmodels.outliers import StahelDonohoOutlyingness
>>> X = np.random.default_rng(0).normal(size=(100, 2))
>>> X[:5] += 6  # five outliers
>>> sdo = StahelDonohoOutlyingness(random_state=0).fit(X)
>>> np.flatnonzero(sdo.predict(X) == -1).tolist()
[0, 1, 2, 3, 4]
>>> sdo.score_samples([[0, 0], [6, 6]]).round(2).tolist()  # projection depth
[0.77, 0.12]

```

| Attribute or method | Meaning |
| --- | --- |
| `outlyingness(X)`, `outlyingness_` | Outlyingness of new and training observations |
| `score_samples(X)` | Projection depth `1 / (1 + outlyingness)`, in [0, 1]; higher is more central, zero for infinite outlyingness |
| `cutoff_`, `offset_` | Outlyingness cutoff and the depth at the cutoff |
| `decision_function(X)` | `score_samples(X) - offset_`; negative for outliers |
| `predict(X)`, `fit_predict(X)` | -1 for outliers, 1 for inliers |
| `directions_` | The unit directions searched |

New observations are scored against the fitted data, which keeps the
directions and the summaries of each projection.

Use [`plots.outlyingness_plot`](plots.md#outlyingness-by-observation) to inspect
the scores, cutoff and flagged counts, including infinite outlyingness.

### Skewed data

On a skewed distribution the long tail inflates a symmetric scale, so
Stahel-Donoho outlyingness can miss outliers on the short side and flag
regular points in the tail. Adjusted outlyingness measures each side with the
whisker of the skewness-adjusted boxplot:

```pycon
>>> from robustmodels.outliers import AdjustedOutlyingness
>>> X = np.random.default_rng(0).lognormal(size=(100, 2))
>>> X[:5] = [-3, -3]  # five outliers on the short side
>>> ao = AdjustedOutlyingness(random_state=0).fit(X)
>>> np.flatnonzero(ao.predict(X) == -1).tolist()
[0, 1, 2, 3, 4]
>>> sdo = StahelDonohoOutlyingness(random_state=0).fit(X)
>>> np.flatnonzero(sdo.predict(X) == -1).tolist()
[37, 39, 82]

```

### Reproducibility

With at most `n_directions` hyperplanes through `p` observations, all of them
are used and the result is deterministic. Otherwise directions are random:
pass `random_state` to get the same result on every run.

## Numerical limits and sources

The outlyingness is the largest standardized projection over a finite set of
directions, normal to hyperplanes through `p` observations (250 times the
number of variables by default, as R's mrfDepth); it can only underestimate
the supremum over all directions. In the tests, on two-dimensional data, it
reaches at least 95% of the supremum found by searching 100,000 directions.

Observations are flagged when their outlyingness exceeds
`exp(median(L) + mad(L) z) - 0.1`, with `L = log(0.1 + outlyingness)` and
`z` the 0.995 normal quantile, as in mrfDepth. It adapts to the data: with no
outliers, few or no points are flagged (under 2% in the tests with normal data).

Adjusted outlyingness computes a medcouple for every direction. Fitting 1000
observations of five variables (1250 directions) takes a few seconds, against
a fraction of a second for Stahel-Donoho outlyingness. Quartiles use the
`(k - 0.5) / n` rule (R's type 5), as mrfDepth.

The observations that define a direction, and any others on the same
hyperplane, have equal projections. Rounding can break those ties, which the
medcouple treats specially, so projections that differ only by rounding are
merged. On the integer stackloss data this reproduces the adjusted
outlyingness computed in exact arithmetic to `1e-14`; mrfDepth, which does not
merge them and whose medcouple departs from its definition for tied data,
differs by up to 14% there, although both flag the same observations. If more
than half of the projections on a direction coincide, the other observations
have infinite outlyingness and are flagged.

Sources: [Stahel-Donoho outlyingness](references.md#s81),
[projection depth](references.md#z03),
[adjusted outlyingness](references.md#hvdv08) and
[the adjusted boxplot](references.md#hv08).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.outliers

.. autoclass:: StahelDonohoOutlyingness
   :members: fit, outlyingness, score_samples, decision_function, predict
.. autoclass:: AdjustedOutlyingness
   :members: fit, outlyingness, score_samples, decision_function, predict
```
