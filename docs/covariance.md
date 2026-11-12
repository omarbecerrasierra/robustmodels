# Robust covariance and multivariate outliers

Robust estimators of multivariate location and scatter, in
`robustmodels.covariance`.

## Choosing an estimator

| Class | Needs | Random | Notes |
| --- | --- | --- | --- |
| `MCD` | `n > p` | Optional | The standard choice; `algorithm="deterministic"` avoids random starts |
| `MRCD` | Any `n`, `p` | No | Regularized MCD for many variables, including `p > n` |
| `MVE` | `n > p + 1` | Yes | Smallest covering ellipsoid; less efficient than MCD |
| `OGK` | `n >= 2` | No | Fast for many variables; not rotation equivariant |
| `SEstimator` | `n > p` | Yes | Smooth weights instead of a hard subset |
| `MMEstimator` | `n > p` | Yes | S-estimator refined to 95% shape efficiency |

Here `n` is the number of observations and `p` the number of variables. The exact MCD, MVE, S and MM definitions are affine equivariant under their
respective assumptions. Numerical searches may behave differently; in
particular deterministic starts do not guarantee full affine equivariance.
Maximal breakdown approaches 50% for fixed dimension as sample size grows,
and can be substantially lower in small samples or high dimensions.

## Use

All classes are scikit-learn estimators: parameters go to the constructor,
`fit(X)` returns the estimator, and results are attributes ending in `_`.

```pycon
>>> import numpy as np
>>> from robustmodels.covariance import MCD
>>> X = np.random.default_rng(0).normal(size=(100, 2))
>>> X[:10] += 10  # ten outliers
>>> mcd = MCD(random_state=0).fit(X)
>>> mcd.location_.round(2).tolist()
[0.01, 0.1]
>>> mcd.covariance_.round(2).tolist()
[[0.93, 0.15], [0.15, 1.06]]

```

Every fitted estimator has:

| Attribute or method | Meaning |
| --- | --- |
| `location_` | Robust location |
| `covariance_` | Robust covariance matrix |
| `precision_` | Its (pseudo-)inverse |
| `dist_` | Squared robust distances of the training data |
| `mahalanobis(X)` | Squared robust distances of any observations |
| `score(X)`, `error_norm(...)`, `get_precision()` | As in `sklearn.covariance.EmpiricalCovariance` |

Estimators that select observations (`MCD`, `MRCD`, `MVE`, `OGK`) also have
`support_`, a boolean mask of the observations used.

### Detecting outliers

Under the normal model, squared robust distances follow approximately a
chi-square distribution with `p` degrees of freedom. Flag the observations
above a high quantile:

```pycon
>>> from scipy.stats import chi2
>>> flagged = mcd.mahalanobis(X) > chi2.ppf(0.975, df=2)
>>> np.flatnonzero(flagged).tolist()
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 69]

```

The ten planted outliers are found. Observation 69 is a clean point in the
tail. The 2.5% tail probability is a population/asymptotic reference; fitted
in-sample distances do not provide an exact finite-sample test or an exact
2.5% false-positive rate.

Use [`plots.distance_distance`](plots.md#distance-distance-plot) to compare
robust and classical distances, or
[`plots.tolerance_ellipse`](plots.md#tolerance-ellipses) for two variables.

### Reproducibility

`MCD` (with `algorithm="fast"`), `MVE`, `SEstimator` and `MMEstimator` draw
random subsets. Pass `random_state` to get the same result on every run, or use
a deterministic estimator:

```pycon
>>> det = MCD(algorithm="deterministic").fit(X)
>>> int(det.support_[:10].sum())  # no outlier is used
0

```

### More variables than observations

`MCD` needs more observations than variables. `MRCD` regularizes the covariance
and works in any dimension:

```pycon
>>> from robustmodels.covariance import MRCD
>>> wide = np.random.default_rng(0).normal(size=(20, 50))
>>> mrcd = MRCD().fit(wide)
>>> mrcd.covariance_.shape
(50, 50)
>>> 0 < mrcd.rho_ < 1  # regularization weight
True

```

### With scikit-learn

The estimators work with `clone`, `get_params`, `set_params` and pipelines:

```pycon
>>> from sklearn.base import clone
>>> clone(mcd).set_params(alpha=0.75).fit(X).h_
75

```

## Numerical limits and sources

MCD is solved by an exhaustive sorted-window search in one dimension.
Multivariate MCD, MRCD, MVE, S and MM use finite searches or iterative
refinements: they do not certify a global optimum. The MVE implementation
searches elemental ellipsoids and local refits; it also does not solve a
minimum-volume enclosing-ellipsoid problem for each selected subset.

For exact raw MCD on data in general position, the finite-sample breakdown
fraction is `min(n-h+1, h-p)/n`; it is not simply 50% for every `n,p`.
Finite-sample correction factors are empirical approximations.

MRCD applies the common-rho rule in Boudt et al., algorithm step 3.4. The
`max_condition` parameter calibrates **initial** subsets in standardized
coordinates. Keeping rho fixed during C-steps does not impose a hard bound
on the final subset's condition number. In original units, unequal column
scales can further increase the condition number.

MM's `efficiency` targets asymptotic **shape** efficiency. Its scale comes from
the S-estimator; 95% shape efficiency does not imply 95% efficiency for location
or all covariance entries. The final normalized bisquare loss must not exceed
the initial loss pointwise (final tuning at least initial tuning). Parameter
combinations violating this condition raise `ValueError`.

Sources: [FAST-MCD](references.md#rvd99), [DetMCD](references.md#hrv12),
[MCD corrections](references.md#pvaw02), [MRCD](references.md#brvv20),
[MVE](references.md#var09), [OGK](references.md#mz02),
[S-estimation](references.md#d87), and [multivariate MM](references.md#sbvaw06).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.covariance

.. autoclass:: MCD
   :members: fit
.. autoclass:: MRCD
   :members: fit
.. autoclass:: MVE
   :members: fit
.. autoclass:: OGK
   :members: fit
.. autoclass:: SEstimator
   :members: fit
.. autoclass:: MMEstimator
   :members: fit
```

### Correction factors

```{eval-rst}
.. autofunction:: h_alpha_n
.. autofunction:: mcd_consistency
.. autofunction:: mcd_small_sample
```
