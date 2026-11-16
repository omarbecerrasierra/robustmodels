# Robust generalized linear models

Use logistic models for two classes and Poisson models for nonnegative count
responses. All four estimators expose `coef_`, `intercept_`, robustness
`weights_`, `cov_params_`, `std_errors_`, normal `z_values_` and `p_values_`,
and an R-style `summary()`. Logistic estimators also provide `classes_`,
`predict_proba` and `decision_function`. Their `predict` returns class labels;
Poisson `predict` returns the fitted mean.

| Estimator | Criterion | Protection |
| --- | --- | --- |
| `MallowsLogisticRegression` | Huberized, Fisher-corrected quasi-score | Bounded residual influence |
| `MallowsPoissonRegressor` | The same score for Poisson means | Bounded residual influence |
| `BYLogisticRegression` | Bianco–Yohai corrected deviance | Large deviances; leverage trimming with `weighted=True` |
| `MTPoissonRegressor` | Bisquare loss of transformed counts | Zero weight for gross residual outliers |

Mallows uses maximum likelihood as its starting fit. `x_weights="mcd"` also
bounds leverage influence using robust predictor distances; it needs continuous
predictors with nonsingular robust scatter. `x_weights="hat"` uses ordinary
leverage. BY defaults to trimming observations beyond the 97.5% robust distance
cutoff before fitting. MT uses `n_trials` random starting subsets, so set
`random_state` for reproducibility. These methods find local solutions and do
not certify global optimality.

## Example

```pycon
>>> import numpy as np
>>> from robustmodels.glm import MallowsPoissonRegressor
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(200, 1))
>>> y = rng.poisson(np.exp(1 + 0.5 * X[:, 0]))
>>> y[:10] = 40
>>> fit = MallowsPoissonRegressor().fit(X, y)
>>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
(1.07, [0.54])
>>> bool((fit.weights_[:10] < 0.2).all())
True
>>> fit.predict(X[:3]).shape
(3,)

```

## Defining equations and inference assumptions

Mallows solves a Huberized Pearson estimating equation with its conditional
model expectation subtracted. For a fixed design, the covariance uses the
conditional score variance and sensitivity for each observation. Independent
tests evaluate Poisson expectations directly and differentiate them numerically;
BY tests verify that its corrected binary score has zero expectation at the
nominal probability. [Cantoni and Ronchetti](references.md#cr01) and
[Croux and Haesbroeck](references.md#ch03) provide the defining theory.

`MTPoissonRegressor` minimizes a transformed-count criterion with ordinary
bisquare loss. Its ideal correction `m(mu)` minimizes
`E[rho(sqrt(Y) - m)]` for Poisson `Y`; the code approximates that function by
a cubic Hermite spline and a continuous value extension above mean 100.
The first grid interval is searched explicitly, because an interior minimum
near zero can be missed by testing only interior grid nodes. Direct Poisson
summation checks the correction equation at small means, rather than using
another program's output as its target.

This is an **MT-style computational variant**. The consistency argument
requires a uniquely defined, strictly increasing correction, correct model
specification and identification. The specific smooth-loss family in
[Valdora and Yohai, section 4](references.md#vy14) uses exponent 4, whereas
ordinary bisquare uses exponent 3. The source theorem's differentiability
assumptions therefore do not automatically apply to this implementation.
`c<=1` is rejected because the correction can remain flat at zero for small
means. `c>1` is a necessary guard here, not a proof of uniqueness, global
monotonicity or every asymptotic assumption for all accepted constants.
The spline has approximation error; local optimization does not certify a
global minimum. Standard errors require a locally identified regular solution.

## Numerical conventions and software correspondence

- `x_weights="hat"` uses `sqrt(1 - h_ii)`, as the Mallows source; robustbase
  uses `(1 - h_ii)^2` under that name. The conditional Mallows sandwich also
  uses an observation-specific mean correction rather than an averaged one.
- Weighted BY fits the trimmed subset. robustbase's `WBY` uses trimming for
  initialization and then minimizes an unweighted criterion. These are
  different procedures, and numerical equality is not an acceptance condition.
- Linear predictors used to evaluate means are clipped to `[-30, 30]` for
  numerical stability. Mallows scoring steps change each predictor by at most
  5. The MT gradient differentiates its implemented clipped objective. A final
  MT solution touching the numerical cap is reported with `converged_=False`,
  a `RuntimeWarning` and unavailable covariance, standard errors and tests
  (`NaN`), so a numerical plateau is not presented as ordinary inference.
- Separated logistic data can lack finite coefficients. Check `converged_`
  and the emitted warnings. A small gradient alone does not establish existence
  or identification. The usual normal inference is asymptotic.
- An all-zero Poisson response with an intercept has no finite intercept and
  raises `ValueError`. Collinear designs and samples without residual degrees
  of freedom are also rejected.

The [validation notes](validation.md) document independent checks, corrected
counterexamples and the remaining scientific limits. Software correspondence
is collected in the [migration guide](from-r.md).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.glm

.. autoclass:: MallowsLogisticRegression
   :members: fit, predict, predict_proba, decision_function, summary
.. autoclass:: MallowsPoissonRegressor
   :members: fit, predict, summary
.. autoclass:: BYLogisticRegression
   :members: fit, predict, predict_proba, decision_function, summary
.. autoclass:: MTPoissonRegressor
   :members: fit, predict, summary
```
