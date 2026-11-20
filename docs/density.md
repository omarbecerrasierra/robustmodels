# Multivariate density estimation

Kernel density estimation with a full bandwidth matrix, in
`robustmodels.density`. The estimate at `x` averages normal densities centred
at the observations, `f(x) = mean_i phi_H(x - X_i)`. The bandwidth matrix `H`
sets the amount and the orientation of the smoothing. A full `H` can smooth
along the directions of correlated data, which a scalar or diagonal bandwidth
cannot. It is selected from the data for the density or for one of its
derivatives.

## Use

```pycon
>>> import numpy as np
>>> from robustmodels.density import KernelDensity
>>> rng = np.random.default_rng(1)
>>> X = np.vstack(
...     [
...         rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 250),
...         rng.multivariate_normal([3, 3], [[0.4, -0.2], [-0.2, 0.4]], 150),
...     ]
... )
>>> kde = KernelDensity().fit(X)  # plug-in bandwidth
>>> kde.bandwidth_.round(3).tolist()
[[0.109, 0.074], [0.074, 0.104]]
>>> np.exp(kde.score_samples([[0, 0], [3, 3], [1.5, 1.5]])).round(3).tolist()
[0.151, 0.083, 0.031]

```

`score_samples` returns the log density, as scikit-learn's density
estimators do. A highest density region of probability `p` is the smallest
region with that probability: the set `{x : f(x) >= level}`. `hdr_level`
estimates its level by a quantile of the estimate at the observations
(Hyndman 1996). Separate regions of the same probability show separate modes.

```pycon
>>> kde.hdr_level([0.25, 0.5, 0.75]).round(4).tolist()
[0.0975, 0.074, 0.046]

```

```{eval-rst}
.. plot::
   :filename-prefix: hdr
   :include-source: true
   :alt: Filled 25%, 50% and 75% highest density regions of a two-mode sample.

   import numpy as np
   from robustmodels import plots
   from robustmodels.density import KernelDensity

   rng = np.random.default_rng(1)
   X = np.vstack([
       rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 250),
       rng.multivariate_normal([3, 3], [[0.4, -0.2], [-0.2, 0.4]], 150),
   ])
   plots.hdr_plot(KernelDensity().fit(X))
```

## Density derivatives

`gradient` and `hessian` differentiate the estimate. Derivatives need more
smoothing than the density: with `deriv_order=1` or `2` the bandwidth is
selected for the gradient or the Hessian (Chacón & Duong 2013). The Hessian
is negative definite near a mode and has eigenvalues of both signs at a
saddle between modes, which locates modes and the valleys between them.

```pycon
>>> hessian = KernelDensity(deriv_order=2).fit(X).hessian([[0, 0], [1.5, 1.5]])
>>> [np.linalg.eigvalsh(h).round(3).tolist() for h in hessian]
[[-0.461, -0.083], [-0.042, 0.073]]

```

## Comparing two samples

`kde_permutation_test` compares two independent samples with a common
bandwidth. It integrates the squared difference of their Gaussian KDEs by
evaluating Gaussian kernels with twice that bandwidth matrix. This is a
squared [maximum mean discrepancy](references.md#gbrss12). Under the null,
the pooled observations are exchangeable. The bandwidth is selected from
the pooled data and held fixed as the labels are reassigned.

```pycon
>>> from robustmodels.density import kde_permutation_test
>>> kde_permutation_test([-2, -1, 0, 1], [2, 3, 4, 5], n_resamples=np.inf).pvalue
0.02857142857142857

```

When the budget covers all assignments, the p-value is exact up to
floating-point precision. With random assignments, the observed assignment
is included in the count. Pass a bandwidth chosen independently of the
labels to use a fixed matrix, including when the pooled covariance is
singular. The kernel matrix and each statistic use quadratic resources in
the pooled sample size.

### Comparing at several scales

`kde_multiscale_test` uses the same base bandwidth and compares the samples
at several smoothing scales. A multiplier `s` uses the matrix `s²H`.
Each scale's discrepancy is standardized using the same label assignments.
The test ranks the observed maximum against the maxima of those assignments,
so the p-value accounts for comparing at several scales. The returned
statistic is this standardized maximum and is dimensionless.

```pycon
>>> from robustmodels.density import kde_multiscale_test
>>> result = kde_multiscale_test([-2, -1, 0, 1], [2, 3, 4, 5], n_resamples=np.inf)
>>> round(result.pvalue, 4)
0.0286

```

The function counts ties in the upper tail. With a random permutation budget,
it includes the observed assignment when standardizing each scale and when
computing the p-value. Choose the bandwidth and the scales independently of
the labels. The joint calibration applies to two independent samples under the
null of a common distribution. Each additional scale adds a quadratic pass
per assignment. One Gram matrix is reused across scales.

The following pilot compares the public single-bandwidth and multiscale
functions. Each scenario used 300 independent pairs of samples of sizes 35
and 45 in two dimensions, with 199 shared random assignments per replicate
and a nominal level of 5%. Both functions used their default bandwidths and
the multiscale function used its five default scales.

| Data for the second sample | Single bandwidth | Multiscale |
| --- | ---: | ---: |
| Both samples standard normal | 22/300 | 19/300 |
| Shift of 0.65 in the first coordinate | 107/300 | 161/300 |
| Twice the standard deviation in the second coordinate | 170/300 | 135/300 |
| Bimodal first coordinate with the same population mean and variance | 142/300 | 184/300 |

The pilot used `np.random.default_rng(3219)` for all four scenarios in order.
In each replicate it drew the first and then the second sample from the
standard normal, applied the change in the table, and drew one shared seed
with `int(rng.integers(2**32))`. For bimodality it multiplied the second
sample's first coordinate by 0.25 and added independent signs from
`rng.choice([-1, 1], size=45)` times `sqrt(1 - 0.25**2)`. The exact binomial
95% intervals under the null are 4.65–10.89% and 3.86–9.71%. The relative
power varies across these alternatives.

### Normal calibration

`kde_test` estimates the population integrated squared difference with
separate bandwidths (Duong, Goud & Schauer 2012). It can detect differences
in location, spread, shape or number of modes.

```pycon
>>> from robustmodels.density import kde_test
>>> one_mode = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 300)
>>> result = kde_test(X, one_mode)
>>> round(result.z, 2), round(result.pvalue, 4)
(3.43, 0.0003)

```

The p-value comes from an asymptotic normal approximation. The statistic
can be negative when the bandwidths differ. The test raises an error when
the variance used for calibration is numerically zero. For finite-sample
calibration under exchangeability, use `kde_permutation_test`.

## Choosing the bandwidth

- **`"plugin"`** (default) minimizes an estimate of the asymptotic mean
  integrated squared error over all positive definite matrices (Chacón &
  Duong 2010). It is the usual choice.
- **`"scv"`**, smoothed cross-validation (Duong & Hazelton 2005; Chacón &
  Duong 2011), replaces the asymptotic bias by the exact bias of
  presmoothed data. It tends to smooth more, which helps when the plug-in
  estimate looks noisy.
- **A number** `h` gives `H = h^2 I`, and **a matrix** is used as `H`.

Both selectors optimize their criteria on the sphered data with a
normal-reference pilot. The resulting selectors are affine equivariant:
transforming the data by a matrix `A` transforms `H` into
`A H A'`. They sum over all pairs of observations, so they take `O(n^2)`
time; a few thousand observations take seconds.

## Numerical checks and sources

The derivatives of the normal kernel needed by the selectors are evaluated
through moments of quadratic forms of a normal vector, computed exactly
from their cumulants, instead of the vectors of all partial derivatives used
by R's ks. The tests compare these moments with Hermite polynomials in one
dimension and with exact Gauss–Hermite quadrature in two. The estimate
integrates to one, its gradient and Hessian agree with finite differences,
and the permutation statistic agrees with independent quadrature of the
squared density difference. Its p-value agrees with enumeration of all
assignments for a small sample with unequal group sizes and ties. Both
selectors are affine equivariant.

On frozen data, R's ks 1.15 gives the same density, gradient, Hessian and
highest density levels (to 1e-12), and the same two-sample statistic and
p-value for the same bandwidths. Its `Hpi` and `Hscv` with
`pilot = "dscalar"` and no binning give the same pilot bandwidths, and
bandwidth matrices within 1e-3. The criterion is lower at our matrices,
because ks stops its search earlier. Other differences from ks:

- ks's default pilot for the density of two variables is the SAMSE pilot,
  and ks bins large samples; robustmodels uses the `dscalar` pilot in every
  case and no binning.
- In `kde_test`, ks evaluates the density gradient at the sample means by
  interpolating a grid estimate; robustmodels evaluates it exactly. ks
  chooses each bandwidth by minimizing a mean squared error whose minimizers
  form a continuum; robustmodels takes the unique minimizer proportional to
  the sample covariance.
- In one dimension ks uses the univariate selectors `hpi` and `hscv`;
  robustmodels uses the same matrix selectors in every dimension.

Sources are Wand & Jones (1993), Chacón & Duong (2010, 2011, 2013),
Chacón, Duong & Wand (2011), Duong & Hazelton (2005), Hyndman (1996) and
Duong, Goud & Schauer (2012); see the [method references](references.md).

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.density

.. autoclass:: KernelDensity
   :members: fit, score_samples, gradient, hessian, hdr_level

.. autofunction:: plugin_bandwidth
.. autofunction:: scv_bandwidth
.. autofunction:: kde_test
.. autofunction:: kde_permutation_test
.. autofunction:: kde_multiscale_test
.. autoclass:: KDETestResult
.. autoclass:: KDEPermutationResult
.. autoclass:: KDEMultiscaleResult
```
