# Robust linear regression

Robust estimators of linear regression coefficients, in
`robustmodels.regression`.

## Choosing an estimator

| Class | Needs | Random | Notes |
| --- | --- | --- | --- |
| `MMRegressor` | `n > p` | Yes | The default choice: 50% breakdown and 95% Gaussian efficiency, as R's `lmrob` |
| `LTSRegressor` | `n > p` | Yes | Least trimmed squares, then reweighted least squares; marks the observations it uses |
| `SRegressor` | `n > p` | Yes | Minimizes a robust residual scale; 50% breakdown but only 29% efficiency |

Here `n` is the number of observations and `p` the number of coefficients,
including the intercept. MM starts from the S-estimator and keeps its
breakdown point while reaching high efficiency, so it is usually preferred to
S alone. LTS is simple to interpret: it fits the `h` observations with the
smallest squared residuals.

## Use

All classes are scikit-learn regressors: parameters go to the constructor,
`fit(X, y)` returns the estimator, and `predict(X)` gives fitted values.

```pycon
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from robustmodels.regression import LTSRegressor
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(100, 1))
>>> y = 1 + 2 * X[:, 0] + rng.normal(scale=0.5, size=100)
>>> y[:20] += 15  # twenty outliers
>>> lts = LTSRegressor(random_state=0).fit(X, y)
>>> round(lts.intercept_, 2), lts.coef_.round(2).tolist()
(0.95, [2.04])
>>> ols = LinearRegression().fit(X, y)
>>> round(float(ols.intercept_), 2), ols.coef_.round(2).tolist()
(4.04, [1.17])

```

The true line is `1 + 2x`. Least squares is pulled towards the outliers; the
robust fits follow the bulk of the data:

```pycon
>>> from robustmodels.regression import MMRegressor
>>> mm = MMRegressor(random_state=0).fit(X, y)
>>> round(mm.intercept_, 2), mm.coef_.round(2).tolist()
(0.96, [2.04])

```

Every fitted estimator has:

| Attribute or method | Meaning |
| --- | --- |
| `coef_`, `intercept_` | Final coefficients |
| `scale_` | Robust scale of the residuals |
| `predict(X)`, `score(X, y)` | Fitted values and R², as in scikit-learn |

`LTSRegressor` also has `support_`, the observations used for the final fit,
the least trimmed squares fit (`raw_coef_`, `raw_intercept_`, `raw_scale_`,
`raw_support_`) and the subset size `h_`. `SRegressor` and `MMRegressor` have
`n_iter_`, and `MMRegressor` keeps its starting S fit in `s_coef_` and
`s_intercept_`; its `scale_` is the S scale.

### Standard errors and tests

`MMRegressor` estimates the covariance matrix of its coefficients with the
formula of Croux, Dhaene & Hoorelbeke (2003), the default of R's `lmrob`, and
reports t statistics and two-sided p-values with `n - p` degrees of freedom.
`summary()` prints them as `summary(lmrob)`:

```pycon
>>> print(mm.summary(["x"]))
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      0.9607     0.05451    17.626  3.77e-32
x                 2.035     0.04985    40.826  2.51e-63
Residual scale: 0.682

```

The values are also attributes: `cov_params_`, `std_errors_`, `t_values_` and
`p_values_`, ordered as `[intercept_, *coef_]`. These standard errors remain
valid when the error variance changes with `X` (heteroskedasticity); when it
does not, they are somewhat less efficient than classical ones. In the tests,
95% intervals cover the true coefficients about 94% of the time with normal
errors and 95% with heteroskedastic errors (`n = 100`).

### Detecting outliers

Divide the residuals by `scale_` and flag those above 2.24 in absolute value,
the cutoff used by the reweighting step (the 98.75% normal quantile):

```pycon
>>> flagged = np.abs(y - lts.predict(X)) > 2.24 * lts.scale_
>>> np.flatnonzero(flagged).tolist() == list(range(20))
True

```

The twenty planted outliers are found. Residuals reveal vertical outliers and
bad leverage points. An observation that is extreme in `X` but follows the line
(a good leverage point) has a small residual; robust distances of `X`, for
example from `robustmodels.covariance.MCD`, identify it.

### Reproducibility

FAST-LTS and FAST-S draw random starting subsets. Pass `random_state` to get
the same result on every run.

### With scikit-learn

The estimators work with `clone`, `get_params`, `set_params`, pipelines and
model selection tools:

```pycon
>>> from sklearn.base import clone
>>> clone(lts).set_params(alpha=0.75).fit(X, y).h_
75

```

Here `alpha` is the coverage of the `h`-subset, as in `MCD` and R's `ltsReg`,
not a regularization strength.

## Numerical limits and sources

FAST-LTS takes two concentration steps from each random elemental start, then
iterates the ten best to convergence. It does not certify a global optimum;
on the classical datasets in the tests it matches robustbase's exhaustive
search. The intercept is a column of ones, without the intercept adjustment
step of Rousseeuw & Van Driessen (2006), as in `ltsReg`'s default.

For data in general position the raw estimator has breakdown point close to
`(n - h + 1) / n`, about 50% at `alpha = 0.5`. The scale uses consistency
factors for normal errors and the small-sample factors of Pison, Van Aelst &
Willems (2002), which are empirical approximations. The consistency factors
assume that the trimmed observations are normal tails: with outliers present,
`scale_` overestimates the error standard deviation (0.73 instead of 0.5 in the
example above), as in `ltsReg`.

FAST-S refines every random start a few steps and the five best to
convergence; refinement stops when the coefficients change by less than `1e-7`
relative, as in `lmrob`. On hbk it finds a smaller S scale than `lmrob`'s
default search, which refines two candidates. The S scale divides by `n - p`,
as `lmrob`, to reduce small-sample bias, and is consistent at the normal model;
like the LTS scale, it overestimates the error standard deviation when outliers
are present. The bisquare constants are computed exactly (1.5476 for 50%
breakdown, 4.6851 for 95% efficiency); `lmrob` rounds them, so scales differ by
about `3e-6`.

The MM-estimator keeps the S scale fixed and converges to a local minimum of
its loss near the S fit. Its breakdown point is that of the S-estimator, and
95% is the asymptotic efficiency of the coefficients under normal errors; the
tests check both efficiencies by simulation.

Sources: [LTS](references.md#r84), [FAST-LTS](references.md#rvd06),
[small-sample corrections](references.md#pvaw02),
[S-estimators](references.md#ry84), [FAST-S](references.md#sby06),
[MM-estimators](references.md#y87), [the `n - p` scale](references.md#mmys19) and
[MM standard errors](references.md#cdh03).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.regression

.. autoclass:: MMRegressor
   :members: fit, predict
.. autoclass:: LTSRegressor
   :members: fit, predict
.. autoclass:: SRegressor
   :members: fit, predict
```
