# Rank-based regression and exact rank tests

Rank-based methods in `robustmodels.rank`: R-estimation of the linear model
with Wilcoxon scores, its drop in dispersion test, and exact Wilcoxon tests
that condition on ties.

## Rank regression

`RankRegressor` minimizes Jaeckel's dispersion of the residuals, a weighted
sum of the residuals by their ranks. It is equivalent to least absolute
deviations on all pairwise differences, so it needs no tuning constant and no
scale estimate, and its fit is the exact minimum of a linear program. Its
Gaussian efficiency is 95.5% and it is more efficient than least squares for
heavy-tailed errors. It resists outliers in the response, not leverage points
in the predictors: for those, use [MM or LTS regression](regression.md).

```pycon
>>> import numpy as np
>>> from robustmodels.rank import RankRegressor
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(100, 2))
>>> y = 1 + X @ [2.0, -1.0] + rng.standard_t(3, size=100)
>>> y[:5] += 20  # five outliers in the response
>>> fit = RankRegressor().fit(X, y)
>>> round(fit.intercept_, 2), fit.coef_.round(2).tolist()
(0.85, [2.14, -1.17])
>>> print(fit.summary())
               Estimate  Std. Error   t value  Pr(>|t|)
(Intercept)      0.8524      0.1747     4.879  4.17e-06
x0                2.138      0.1297    16.486  6.95e-30
x1               -1.168      0.1331    -8.774  5.92e-14
Drop in dispersion test: F = 97.89 on 2 and 97 degrees of freedom, p-value 5.39e-24
Robust R-squared: 0.6687

```

Least squares gives `(1.95, 2.3, -1.34)` for the same data. The standard
errors use the scale `tau` of Koul, Sievers and McKean (1987) for the slopes
and `tau_s`, from the confidence interval of the median, for the intercept.
For large samples the `tau` calculation selects its quantile and counts
distances in sorted rows with linear auxiliary memory. The slope fit starts
from Rfit's BFGS minimization of the dispersion, then solves the linear
program only on the pairs whose residuals are close; the other pairs keep
their signs and enter as one linear term ([Portnoy and Koenker](references.md#pk97)).
The solution is accepted only when no excluded pair can change sign, so the
fit is still the exact minimum: a sample of 2,000 takes about a second
instead of minutes. With group indicators the minimum is often not unique
(as the median of an even number of values); the fit is then the minimizer
with the smallest residual sum of squares, so `tau`, `drop_test` and
`rank_anova` do not depend on the group labels or the contrasts.
`drop_test(full, reduced)` compares nested fits on the same response, the rank
analogue of the F test; `summary()` reports it for all slopes, with the robust
R² of Rfit. `rank_anova(y, groups)` is the one-way analysis of variance by
ranks (Rfit's `oneway.rfit`): the drop test of the model with one center per
group.

The drop test checks that the response is the same in the same row order
and that the reduced design lies in the full design's column space,
including the intercept. Its F calibration is approximate under independent,
identically distributed continuous errors. Inference that requires an
unidentified residual scale is reported as NaN. With one residual degree
of freedom, the small-sample factor for `tau_s` uses a denominator of one
in place of the zero in Rfit's correction.

```pycon
>>> from robustmodels.rank import rank_anova
>>> groups = np.repeat(["a", "b", "c"], 15)
>>> y3 = rng.standard_t(3, size=45) + np.repeat([0.0, 0.0, 1.5], 15)
>>> rank_anova(y3, groups).df
(2, 42)

```

An exact Kruskal-Wallis test conditional on ties would need the joint
distribution of the rank sums of all groups but one; R's coin and
exactRankTests offer exact rank tests only for one and two samples. For small
samples, `scipy.stats.permutation_test` enumerates all assignments. The residual diagnostics of [`plots`](plots.md) accept a fitted
`RankRegressor`, whose `scale_` is the MAD of the residuals.

## Exact tests with ties

`rank_sum_test(x, y)` (Wilcoxon–Mann–Whitney) and `signed_rank_test(x, y)`
(one-sample or paired Wilcoxon) compute exact p-values from the permutation
distribution of the observed midranks. That distribution depends on the ties;
tables or algorithms that assume no ties give other p-values:

```pycon
>>> from scipy import stats
>>> from robustmodels.rank import rank_sum_test, signed_rank_test
>>> x, y = [1, 2, 2, 3, 3, 3, 5], [2, 3, 4, 4, 5, 6]
>>> round(rank_sum_test(x, y).pvalue, 4)
0.1166
>>> round(float(stats.mannwhitneyu(x, y, method="exact").pvalue), 4)  # ignores the ties
0.1375
>>> d = [0.5, -1, 1, 2, 2, -0.5, 3, 1, 0, 1.5]
>>> round(signed_rank_test(d).pvalue, 4), round(signed_rank_test(d, zero_method="pratt").pvalue, 4)
(0.0508, 0.0547)

```

The two-sided p-value of the rank sum counts the assignments at least as far
from the mean as the observed one, as R's coin and exactRankTests; the
signed-rank distribution is symmetric, so both conventions agree there. Zeros
are dropped (`"wilcox"`, R's default) or ranked without a sign (`"pratt"`,
coin's default). The distributions are computed on doubled midranks, which
are integers, so ties compare exactly. Computing the rank sum distribution
costs about `m N^3` operations for samples of sizes `m <= n` and `N = m + n`.

For paired signed ranks the null shift `mu` is subtracted from `x - y`. The
test sums the three terms accurately so that a small shift remains visible
when two large terms cancel. If the final difference overflows float64,
the test raises `ValueError`. Rescale both samples and `mu` to common units
before testing in that case.

## Numerical checks and sources

- Exact p-values equal the enumeration of every assignment or sign pattern in
  small samples with ties and zeros, all three alternatives; they also equal
  scipy's exact tests without ties and R's `exactRankTests::wilcox.exact` and
  coin's exact `wilcox_test` and `wilcoxsign_test` on frozen tied data.
- The fit is the exact minimum of the dispersion: random perturbations never
  lower the pairwise loss, and with one predictor it equals the weighted
  median of the pairwise slopes. Rfit's quasi-Newton minimum is never lower
  and agrees to about 1e-3 in the coefficients.
- `tau` takes the 0.8 quantile of the pairwise residual differences exactly,
  where Rfit's default routine solves for it numerically to an absolute
  tolerance of 1.2e-4, and its R version can step to a neighbouring order
  statistic through rounding. On Rfit's residuals the two agree to 1e-4.
  The estimates converge to their population values at the normal model.
- Rfit standardizes the scores so that their sum of squares is `n + 1` and
  applies small-sample corrections to `tau` and `tau_s`; these are kept because
  in simulations (`n = 20` and `40`, normal, t3 and contaminated errors) they
  gave 95% intervals and 5% drop tests closer to the nominal levels than the
  unstandardized scores. The test suite checks the coverage and the size under
  t3 errors.

Sources are Jaeckel (1972), McKean and Hettmansperger (1976), Koul, Sievers
and McKean (1987), Hettmansperger and McKean (2011), Kloke and McKean (2012),
Streitberg and Röhmel (1986), Hothorn et al. (2006) and Pratt (1959).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.rank

.. autoclass:: RankRegressor
   :members: fit, predict, summary
.. autofunction:: drop_test
.. autofunction:: rank_anova
.. autofunction:: rank_sum_test
.. autofunction:: signed_rank_test
```
