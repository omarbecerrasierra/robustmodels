# Shape-constrained estimation

Estimators that replace a tuning constant by a shape restriction, in
`robustmodels.shape`: a log-concave density estimate needs no bandwidth, and
regression by antitonic score matching needs no loss function.

## Log-concave densities

A density is log-concave when its logarithm is concave. Normal, logistic,
Laplace, gamma and beta densities with shape at least one, and uniform
densities, all are. Such densities are unimodal with tails at most exponential. Among
them, `LogConcaveDensity` finds the one of largest likelihood (Walther 2002).
The estimate exists for any sample with two distinct values, needs no
bandwidth, and its log density is piecewise linear, with knots at some
observations (Dümbgen & Rufibach 2009).

```pycon
>>> import numpy as np
>>> from robustmodels.shape import LogConcaveDensity
>>> x = np.random.default_rng(3).gamma(2, size=250)
>>> fit = LogConcaveDensity().fit(x)
>>> fit.knots_.round(3).tolist()
[0.145, 0.709, 1.272, 1.327, 1.329, 2.848, 5.934, 8.969]
>>> np.exp(fit.score_samples([0.5, 2, 5])).round(3).tolist()
[0.192, 0.291, 0.041]
>>> fit.cdf([1, 2, 4]).round(3).tolist()
[0.183, 0.528, 0.887]

```

The estimate is zero outside the range of the sample, has the mean of the
sample and at most its variance. When the restriction is in doubt, compare
it with a kernel estimate: two modes or heavier tails contradict it.

```{eval-rst}
.. plot::
   :filename-prefix: logconcave
   :include-source: true
   :alt: Log-concave density estimate with its knots over the histogram of a gamma sample.

   import numpy as np
   from robustmodels import plots
   from robustmodels.shape import LogConcaveDensity

   x = np.random.default_rng(3).gamma(2, size=250)
   plots.log_concave_plot(LogConcaveDensity().fit(x))
```

## Regression by antitonic score matching

An M-estimator with a convex loss has a unique solution that is easy to
compute, but its efficiency depends on how well the loss fits the errors:
least squares for normal errors, least absolute deviations for Laplace
errors. `ScoreMatchingRegressor` estimates the loss from the data (Feng,
Kao, Xu & Samworth 2024). Its derivative is minus the antitonic projected
score: the decreasing function closest to the score `(log p)'` of the error
density `p`. Among convex M-estimators, this loss gives the smallest
asymptotic variance. For log-concave errors it is the maximum likelihood
loss. For heavier tails it is bounded, so it resists outliers in the
response.

```pycon
>>> from robustmodels.shape import ScoreMatchingRegressor
>>> rng = np.random.default_rng(4)
>>> X = rng.normal(size=(400, 2))
>>> y = 1 + X @ [2.0, -1.0] + rng.standard_t(2, 400)  # least squares: (0.69, 2.0, -0.86)
>>> fit = ScoreMatchingRegressor().fit(X, y)
>>> print(fit.summary(["x1", "x2"]))
               Estimate  Std. Error   z value  Pr(>|z|)
(Intercept)      0.9944     0.07473    13.306  2.12e-40
x1                2.103     0.07094    29.641  4.43e-193
x2               -1.021     0.07708   -13.242  5e-40

```

The fit starts from least absolute deviations, estimates the score from its
residuals with a normal kernel density estimate, minimizes the convex loss,
and re-estimates the score once. Standard errors come from
`(information_ X'X)^-1`, with `information_` the mean squared score of the
residuals. With `symmetric=True` (default) the errors are assumed symmetric
and the intercept is estimated with the slopes. With `symmetric=False` the
slopes need no symmetry, and the intercept is the median of the residuals.

```{eval-rst}
.. plot::
   :filename-prefix: score
   :include-source: true
   :alt: Estimated antitonic score, bounded in the tails, against the normal score.

   import numpy as np
   from robustmodels import plots
   from robustmodels.shape import ScoreMatchingRegressor

   rng = np.random.default_rng(4)
   X = rng.normal(size=(400, 2))
   y = 1 + X @ [2.0, -1.0] + rng.standard_t(2, 400)
   plots.score_plot(ScoreMatchingRegressor().fit(X, y), X, y)
```

The score resists outliers in the response, not leverage points; for those,
use `MMRegressor`. With normal errors the fit is close to least squares.

## Numerical checks and sources

The log-concave estimate is checked against its characterization (Dümbgen &
Rufibach 2009): the log density is concave, no kink at an observation
increases the likelihood, the density integrates to one, and its mean is the
sample mean. Two distinct values give the uniform density. The integrals of
the exponential of linear functions match numerical quadrature. On frozen
data R's logcondens 2.1 finds the same knots, and our log-likelihoods are
higher: logcondens stops its Newton steps earlier.

For the regression, the estimated score is decreasing (and odd when
symmetric), the fit solves the estimating equation `X' psi(residuals) = 0`,
and it is regression and scale equivariant. With normal errors it is close
to least squares. On frozen data R's asm 0.2 gives the same coefficients (to
1e-6) and standard errors (to 1e-4); the density estimate of the residuals
is binned on the same grid. asm fails when more than half the residuals are
equal; robustmodels then uses all residuals for the density estimate.

Sources are Walther (2002), Pal, Woodroofe & Meyer (2007), Dümbgen & Rufibach
(2009, 2011), Dümbgen, Hüsler & Rufibach (2007) and Feng, Kao, Xu & Samworth
(2024); see the [method references](references.md).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.shape

.. autoclass:: LogConcaveDensity
   :members: fit, score_samples, cdf

.. autoclass:: ScoreMatchingRegressor
   :members: fit, predict, psi, summary
```
