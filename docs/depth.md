# Depth and the bagplot

Depth measures centrality relative to a sample. `halfspace_depth` is the
smallest fraction of observations in a closed halfspace whose boundary passes
through the query point. It is exact in one and two dimensions, including ties.
Above two dimensions, it searches directions normal to sample hyperplanes and
returns an approximation that can overestimate the exact depth. Supply
`n_directions` and `random_state` to control that search, or `exact=True` for
the exact depth by the recursive algorithm of Dyckerhoff and Mozharovskyi
(2016). It needs no general position and costs `O(n^(p-1) log n)` per point:
the depths of all 200 points of a sample in three dimensions take about 6
seconds on a laptop, and those of 60 points in four dimensions about 12.

```pycon
>>> import numpy as np
>>> from robustmodels.depth import halfspace_depth
>>> X3 = np.random.default_rng(1).normal(size=(30, 3))
>>> exact = halfspace_depth(X3, exact=True)
>>> bool(np.all(halfspace_depth(X3, random_state=0) >= exact))  # the search never underestimates
True

```

`regression_depth` measures the smallest fraction of observations crossed when
tilting a fitted regression plane to a vertical nonfit. It is exact with one
predictor. With several predictors, enumerating all directions is exact for
data in general position; a limited random search can overestimate depth.
Predictors exclude the intercept column. Depths are fractions in `[0, 1]`.

## Bivariate center and outliers

`halfspace_median` returns the area centroid of the deepest bivariate region
(or the midpoint of a segment, or its only point). `bagplot_stats` builds the
bag containing approximately half the sample, its inflated fence, the outlier
mask, and the loop containing the bag and observations inside the fence.
`plots.bagplot` draws these statistics and returns a Matplotlib axes.

```pycon
>>> import numpy as np
>>> from robustmodels.depth import bagplot_stats, halfspace_depth
>>> square = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]
>>> halfspace_depth(square).tolist()
[0.2, 0.2, 0.2, 0.2, 0.6]
>>> X = np.random.default_rng(0).normal(size=(100, 2))
>>> X[:3] = [[6, 6], [7, -6], [-8, 5]]
>>> np.flatnonzero(bagplot_stats(X).outliers).tolist()
[0, 1, 2]

```

```{eval-rst}
.. plot::
   :filename-prefix: bagplot
   :include-source: true
   :alt: Bagplot with a Tukey median, central bag, fence and three outliers.

   import numpy as np
   from robustmodels import plots

   X = np.random.default_rng(0).normal(size=(100, 2))
   X[:3] = [[6, 6], [7, -6], [-8, 5]]
   plots.bagplot(X, feature_names=["First measurement", "Second measurement"])
```

## Comparing two samples

The depth-depth plot (Liu, Parelius & Singh 1999) places every observation of
two samples at its depth in each. Samples from one distribution keep the
points near the diagonal; a difference in location bends them into a lens,
and a difference in spread moves them to one side, as here.
[`density.kde_test`](density.md#comparing-two-samples) tests the difference.

```{eval-rst}
.. plot::
   :filename-prefix: ddplot
   :include-source: true
   :alt: Depth-depth plot of two samples whose spreads differ, with points off the diagonal.

   import numpy as np
   from robustmodels import plots

   rng = np.random.default_rng(4)
   control = rng.normal(size=(150, 2))
   treated = rng.normal(size=(120, 2)) * [1.6, 1]
   plots.dd_plot(control, treated, labels=["Control", "Treated"])
```

## Numerical checks and limits

The bivariate median and bag use a subset of at most `max_samples=200`
observations. Set a seed for larger samples. The complete data must span two
dimensions; a singular selected subset raises `ValueError` and requires a
larger `max_samples`. Duplicated observations are supported, including a bag
that collapses to a segment or a point. The bag boundary interpolates depth
regions over a finite angular grid, so transformed bag vertices need not be
exactly affine equivariant. Univariate halfspace depth compares represented
values directly, so a distinct external point is not merged into the sample.
Bivariate depth evaluates open angular cells after diagonal conditioning;
independent rational-sign enumeration checks narrow-angle cases. Opposite
offsets from the query point, as when it lies between two observations, share
one line angle, and lines closer than `1e-13` radians after the conditioning
are taken as one. Version 0.2.0 compared separately rounded angles instead and
could open a spurious cell: for the query `(0.5, 1.5)` between the observations
`(-1, 0)` and `(2, 3)` of an integer sample it returned `2/9` instead of `3/9`.
Exact depth in three or more dimensions is checked against linear-programming
separation of every subset in small samples and against R's ddalpha.
Projection and region calculations retain numerical tolerances.

The bag interpolates both adjacent depth regions even when the inner one
is a nonzero segment. An explicit tied five-point sample checks its diagonal
boundary at `(13/6, 13/6)`. Small regression-depth problems in three
predictor dimensions are checked by exhaustive deletion and linear-programming
separation. Frozen mrfDepth/aplpack outputs retain their provenance for
supplementary regression checks; invalid captured depths are excluded as
mathematical witnesses. The definition and the independent geometry checks
determine correctness, including when external software differs.

Sources are Tukey (1975), Rousseeuw and Ruts (1996, 1998), Rousseeuw and Hubert
(1999), Rousseeuw, Ruts and Tukey (1999) and Dyckerhoff and Mozharovskyi (2016);
full citations follow.

## Reference

```{eval-rst}
.. currentmodule:: robustmodels.depth

.. autofunction:: halfspace_depth
.. autofunction:: regression_depth
.. autofunction:: halfspace_median
.. autofunction:: bagplot_stats
```
