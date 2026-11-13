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

## Numerical conventions and R

The sources are Cantoni and Ronchetti (2001), Bianco and Yohai (1996), Croux
and Haesbroeck (2003), and Valdora and Yohai (2014); full citations appear in
the class references below.

- Mallows coefficients and residual weights agree with robustbase's `Mqle`
  on the frozen reference samples to about `1e-10`. Its sandwich covariance
  subtracts the correction's squared mean separately for every observation;
  robustbase subtracts an averaged correction. Standard errors can differ.
- `x_weights="hat"` uses `sqrt(1 - h_ii)`, as the paper; robustbase uses
  `(1 - h_ii)^2` under that name.
- Weighted BY fits the trimmed subset. robustbase's `WBY` uses trimming for
  initialization and then minimizes the unweighted criterion. Use
  `weighted=False` for comparisons with its unweighted `BY`.
- MT interpolates the Fisher-consistent transformation on a grid using cubic
  Hermite splines and extends it continuously above mean 100. robustbase's
  interpolation and tail convention differ. Reference coefficients agree to
  about `1e-3`, rather than machine precision.
- Linear predictors used to evaluate means are clipped to `[-30, 30]` for
  numerical stability. Scoring steps change each linear predictor by at most
  5. Separated logistic data can still lack finite coefficients: check
  `converged_` and the emitted `RuntimeWarning`. A small gradient is a local
  optimization criterion, not an existence or identifiability guarantee.
- An all-zero Poisson response with an intercept has no finite intercept and
  raises `ValueError`. Collinear designs and samples with no residual degrees
  of freedom are also rejected. Normal inference is asymptotic.

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
