# Robust linear regression

Robust estimators of linear regression coefficients, in
`robustmodels.regression`.

## Choosing an estimator

| Class | Needs | Random | Notes |
| --- | --- | --- | --- |
| `LTSRegressor` | `n > p` | Yes | Least trimmed squares, then reweighted least squares |

Here `n` is the number of observations and `p` the number of coefficients,
including the intercept. S and MM regression are next on the
[roadmap](roadmap.md).

## Use

`LTSRegressor` is a scikit-learn regressor: parameters go to the constructor,
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

The true line is `1 + 2x`. Least squares is pulled towards the outliers; LTS
fits the bulk of the data. Every fitted `LTSRegressor` has:

| Attribute or method | Meaning |
| --- | --- |
| `coef_`, `intercept_` | Final coefficients |
| `scale_` | Robust scale of the residuals |
| `support_` | Boolean mask of the observations used for the final fit |
| `raw_coef_`, `raw_intercept_`, `raw_scale_`, `raw_support_` | The least trimmed squares fit and its `h`-subset |
| `h_` | Size of the `h`-subset |
| `predict(X)`, `score(X, y)` | Fitted values and R², as in scikit-learn |

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

FAST-LTS draws random starting subsets. Pass `random_state` to get the same
result on every run.

### With scikit-learn

`LTSRegressor` works with `clone`, `get_params`, `set_params`, pipelines and
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

Sources: [LTS](references.md#r84), [FAST-LTS](references.md#rvd06) and
[small-sample corrections](references.md#pvaw02).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.regression

.. autoclass:: LTSRegressor
   :members: fit, predict
```
